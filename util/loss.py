import tensorflow as tf

def bpr_loss(drug_emb,pos_disease_emb,neg_disease_emb):
    score = tf.reduce_sum(tf.multiply(drug_emb, pos_disease_emb), 1) - tf.reduce_sum(tf.multiply(drug_emb, neg_disease_emb), 1)
    loss = -tf.reduce_sum(tf.log(tf.sigmoid(score)+10e-8))
    return loss
