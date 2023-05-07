import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import os
from util.config import ModelConf
from util.io import FileIO
from util.dataSplit import *
from util import config
from util.loss import bpr_loss
from util.rating import Rating
from random import shuffle,choice

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class AntiViralDL():
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        self.config = conf
        self.data = Rating(self.config, trainingSet, testSet)
        self.num_drugs, self.num_diseases, self.train_size = self.data.trainingSize()


    def readConfiguration(self):
        self.batch_size = int(self.config['batch_size'])
        self.emb_size = int(self.config['num.factors'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        learningRate = config.OptionConf(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        self.batch_size = int(self.config['batch_size'])
        regular = config.OptionConf(self.config['reg.lambda'])
        self.regU, self.regI, self.regB = float(regular['-u']), float(regular['-i']), float(regular['-b'])

        args = config.OptionConf(self.config['AntiViralDL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])

    def LightGCN_encoder(self, emb, adj, n_layers):
        all_embs = []
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb)
            all_embs.append(emb)
        all_embs = tf.reduce_mean(all_embs, axis=0)
        return tf.split(all_embs, [self.num_drugs, self.num_diseases], 0)

    def perturbed_LightGCN_encoder(self, emb, adj, n_layers):
        all_embs = []
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb)
            random_noise = tf.random.uniform(emb.shape)
            emb += tf.multiply(tf.sign(emb), tf.nn.l2_normalize(random_noise, 1)) * self.eps
            all_embs.append(emb)
        all_embs = tf.reduce_mean(all_embs, axis=0)
        return tf.split(all_embs, [self.num_drugs, self.num_diseases], 0)

    def initModel(self):
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.r = tf.placeholder(tf.float32, name="rating")
        self.drug_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_drugs, self.emb_size], stddev=0.005),
                                           name='U')
        self.disease_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_diseases, self.emb_size], stddev=0.005),
                                           name='V')
        self.batch_drug_emb = tf.nn.embedding_lookup(self.drug_embeddings, self.u_idx)
        self.batch_pos_disease_emb = tf.nn.embedding_lookup(self.disease_embeddings, self.v_idx)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        initializer = tf.contrib.layers.xavier_initializer()
        self.drug_embeddings = tf.Variable(initializer([self.num_drugs, self.emb_size]))
        self.disease_embeddings = tf.Variable(initializer([self.num_diseases, self.emb_size]))
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        ego_embeddings = tf.concat([self.drug_embeddings, self.disease_embeddings], axis=0)
        # adjaceny matrix
        self.norm_adj = self.create_joint_sparse_adj_tensor()
        # encoding
        self.main_drug_embeddings, self.main_disease_embeddings = self.LightGCN_encoder(ego_embeddings, self.norm_adj,
                                                                                     self.n_layers)
        self.perturbed_drug_embeddings1, self.perturbed_disease_embeddings1 = self.perturbed_LightGCN_encoder(
            ego_embeddings, self.norm_adj, self.n_layers)
        self.perturbed_drug_embeddings2, self.perturbed_disease_embeddings2 = self.perturbed_LightGCN_encoder(
            ego_embeddings, self.norm_adj, self.n_layers)
        self.batch_neg_disease_emb = tf.nn.embedding_lookup(self.main_disease_embeddings, self.neg_idx)
        self.batch_drug_emb = tf.nn.embedding_lookup(self.main_drug_embeddings, self.u_idx)
        self.batch_pos_disease_emb = tf.nn.embedding_lookup(self.main_disease_embeddings, self.v_idx)

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

    def create_joint_sparse_adj_tensor(self):
        norm_adj = self.create_joint_sparse_adjaceny()
        row, col = norm_adj.nonzero()
        indices = np.array(list(zip(row, col)))
        adj_tensor = tf.SparseTensor(indices=indices, values=norm_adj.data, dense_shape=norm_adj.shape)
        return adj_tensor


    def calc_cl_loss(self):
        p_drug_emb1 = tf.nn.embedding_lookup(self.perturbed_drug_embeddings1, tf.unique(self.u_idx)[0])
        p_disease_emb1 = tf.nn.embedding_lookup(self.perturbed_disease_embeddings1, tf.unique(self.v_idx)[0])
        p_drug_emb2 = tf.nn.embedding_lookup(self.perturbed_drug_embeddings2, tf.unique(self.u_idx)[0])
        p_disease_emb2 = tf.nn.embedding_lookup(self.perturbed_disease_embeddings2, tf.unique(self.v_idx)[0])

        normalize_emb_drug1 = tf.nn.l2_normalize(p_drug_emb1, 1)
        normalize_emb_drug2 = tf.nn.l2_normalize(p_drug_emb2, 1)
        normalize_emb_disease1 = tf.nn.l2_normalize(p_disease_emb1, 1)
        normalize_emb_disease2 = tf.nn.l2_normalize(p_disease_emb2, 1)
        pos_score_u = tf.reduce_sum(tf.multiply(normalize_emb_drug1, normalize_emb_drug2), axis=1)
        pos_score_i = tf.reduce_sum(tf.multiply(normalize_emb_disease1, normalize_emb_disease2), axis=1)
        ttl_score_u = tf.matmul(normalize_emb_drug1, normalize_emb_drug2, transpose_a=False, transpose_b=True)
        ttl_score_i = tf.matmul(normalize_emb_disease1, normalize_emb_disease2, transpose_a=False, transpose_b=True)
        pos_score_u = tf.exp(pos_score_u / 0.2)
        ttl_score_u = tf.reduce_sum(tf.exp(ttl_score_u / 0.2), axis=1)
        pos_score_i = tf.exp(pos_score_i / 0.2)
        ttl_score_i = tf.reduce_sum(tf.exp(ttl_score_i / 0.2), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score_u / ttl_score_u)) - tf.reduce_sum(tf.log(pos_score_i / ttl_score_i))

        return self.cl_rate * cl_loss

    def trainModel(self):

        rec_loss = bpr_loss(self.batch_drug_emb, self.batch_pos_disease_emb, self.batch_neg_disease_emb)
        rec_loss += self.regU * (
                    tf.nn.l2_loss(self.batch_drug_emb) + tf.nn.l2_loss(self.batch_pos_disease_emb) + tf.nn.l2_loss(
                self.batch_neg_disease_emb))

        self.cl_loss = self.calc_cl_loss()
        loss = rec_loss + self.cl_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                drug_idx, i_idx, j_idx = batch
                _, l, rec_l, cl_l = self.sess.run([train, loss, rec_loss, self.cl_loss],
                                                  feed_dict={self.u_idx: drug_idx, self.neg_idx: j_idx,
                                                             self.v_idx: i_idx})
                print('training:', epoch + 1, 'batch', n, 'total_loss:', l, 'rec_loss:', rec_l, 'cl_loss', cl_l)
            self.U, self.V = self.sess.run([self.main_drug_embeddings, self.main_disease_embeddings])

    def execute(self):
        self.readConfiguration()
        self.initModel()
        self.trainModel()


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
        model.execute()
