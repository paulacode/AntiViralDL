import tensorflow as tf


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    score = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), 1) - tf.reduce_sum(tf.multiply(user_emb, neg_item_emb), 1)
    loss = -tf.reduce_sum(tf.math.log(tf.sigmoid(score) + 10e-8))
    return loss


def InfoNCE(view1, view2, temperature):
    pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
    ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
    pos_score = tf.exp(pos_score / temperature)
    ttl_score = tf.reduce_sum(tf.exp(ttl_score / temperature), axis=1)
    cl_loss = -tf.reduce_sum(tf.math.log(pos_score / ttl_score))
    return cl_loss

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += tf.norm(emb, ord='euclidean')  
    return emb_loss * reg