import tensorflow as tf
from tensorflow.keras import layers
from util.loss import bpr_loss, l2_reg_loss, InfoNCE  # You'll need to create this file with the equivalent loss functions
from util.config import ModelConf
from util.io import FileIO
from util.rating import Rating
from util import config
from random import shuffle, randint, choice
import numpy as np
import scipy.sparse as sp

class AntiViralDL_Encoder(tf.keras.Model):
    def __init__(self, data, emb_size, eps, n_layers):
        super(AntiViralDL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.num_drugs, self.num_diseases, self.train_size = self.data.trainingSize()
        initializer = tf.keras.initializers.GlorotUniform()
        self.drug_emb = tf.Variable(initializer(shape=[self.num_drugs, self.emb_size]), trainable=True)
        self.disease_emb = tf.Variable(initializer(shape=[self.num_diseases, self.emb_size]), trainable=True)

    def create_joint_sparse_adj_tensor(self):
        norm_adj = self.create_joint_sparse_adjaceny()
        row, col = norm_adj.nonzero()
        indices = np.array(list(zip(row, col)))
        adj_tensor = tf.sparse.SparseTensor(indices=indices, values=norm_adj.data, dense_shape=norm_adj.shape)
        return adj_tensor

    def create_joint_sparse_adjaceny(self):
        n_nodes = self.num_drugs + self.num_diseases
        row_idx = [self.data.drug[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.disease[pair[1]] for pair in self.data.trainingData]
        drug_np = np.array(row_idx)
        disease_np = np.array(col_idx)
        ratings = np.ones_like(drug_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (drug_np, disease_np + self.num_drugs)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def call(self, perturbed=False):
        # print(f'self.drug_emb: {self.drug_emb}')
        ego_embeddings = tf.concat([self.drug_emb, self.disease_emb], axis=0)
        all_embeddings = []
        sparse_norm_adj = self.create_joint_sparse_adj_tensor()
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse.sparse_dense_matmul(sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = tf.random.normal(tf.shape(ego_embeddings))
                ego_embeddings += tf.math.sign(ego_embeddings) * tf.math.l2_normalize(random_noise, axis=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = tf.stack(all_embeddings, axis=1)
        all_embeddings = tf.reduce_sum(all_embeddings, axis=1)
        drug_all_embeddings, disease_all_embeddings = tf.split(all_embeddings, [self.num_drugs, self.num_diseases])
        return drug_all_embeddings, disease_all_embeddings

class AntiViralDL():
    def __init__(self, conf, training_set, test_set):
        self.config = conf
        self.data = Rating(self.config, training_set, test_set)
        self.num_drugs, self.num_diseases, self.train_size = self.data.trainingSize()

        self.batch_size = int(self.config['batch_size'])
        self.emb_size = int(self.config['num.factors'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        learningRate = config.OptionConf(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        regular = config.OptionConf(self.config['reg.lambda'])
        self.regU, self.regI, self.regB = float(regular['-u']), float(regular['-i']), float(regular['-b'])

        args = config.OptionConf(self.config['AntiViralDL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])

        self.model = AntiViralDL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                drugs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                diseases = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                drugs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                diseases = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            disease_list = list(self.data.disease.keys())
            for i, drug in enumerate(drugs):
                i_idx.append(self.data.disease[diseases[i]])
                u_idx.append(self.data.drug[drug])
                neg_disease = choice(disease_list)
                while neg_disease in self.data.trainSet_u[drug]:
                    neg_disease = choice(disease_list)
                j_idx.append(self.data.disease[neg_disease])

            yield u_idx, i_idx, j_idx

    def train(self):
        model = self.model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                with tf.GradientTape() as tape:
                    drug_idx, pos_idx, neg_idx = batch
                    rec_drug_emb, rec_disease_emb = model.call()
                    drug_emb, pos_disease_emb, neg_disease_emb = tf.gather(rec_drug_emb, drug_idx), \
                                                        tf.gather(rec_disease_emb, pos_idx), \
                                                        tf.gather(rec_disease_emb, neg_idx)
                    rec_loss = bpr_loss(drug_emb, pos_disease_emb, neg_disease_emb)
                    cl_loss = self.cl_rate * self.cal_cl_loss([drug_idx, pos_idx])
                    batch_loss = rec_loss + l2_reg_loss(self.regU, drug_emb, pos_disease_emb) + cl_loss
                # Backward and optimize
                gradients = tape.gradient(batch_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print('epoch:', epoch + 1, 'loss', batch_loss.numpy(), 'rec_loss:', rec_loss.numpy(), 'cl_loss', cl_loss.numpy())

    def cal_cl_loss(self, idx):
        u_idx, i_idx = idx[0], idx[1]
        # u_idx = tf.unique(tf.convert_to_tensor(idx[0]), out_idx=True)[0]
        # i_idx = tf.unique(tf.convert_to_tensor(idx[1]), out_idx=True)[0]
        drug_view_1, disease_view_1 = self.model(perturbed=True)
        drug_view_2, disease_view_2 = self.model(perturbed=True)
        drug_emb_1, pos_disease_emb_1 = tf.gather(drug_view_1, u_idx), tf.gather(disease_view_1, i_idx)
        drug_emb_2, pos_disease_emb_2 = tf.gather(drug_view_2, u_idx), tf.gather(disease_view_2, i_idx)

        drug_cl_loss = InfoNCE(drug_emb_1, drug_emb_2, 0.2)
        disease_cl_loss = InfoNCE(pos_disease_emb_1, pos_disease_emb_1, 0.2)
        return drug_cl_loss + disease_cl_loss


if __name__ == '__main__':
    conf = ModelConf('AntiViralDL.conf')
    for i in range(0, 5):
        train_path = f"data/train_{i}.txt"
        test_path = f"data/test_{i}.txt"
        binarized = False
        bottom = 0
        trainingData = FileIO.loadDataSet(conf, train_path, binarized=binarized, threshold=bottom)
        testData = FileIO.loadDataSet(conf, test_path, bTest=True, binarized=binarized,
                                      threshold=bottom)
        model = AntiViralDL(conf, trainingData, testData)
        model.train()
