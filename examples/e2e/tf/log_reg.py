#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
from sgd import define_args, train_with_sgd

import time
import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def build_graph(max_dim, cluster_spec):
    FLAGS = tf.app.flags.FLAGS
    X = tf.sparse_placeholder(tf.float32, shape=[None, max_dim], name="x-input")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")
    W = tf.Variable(tf.zeros(shape=[max_dim, 1]))
    b = tf.Variable(tf.zeros(shape=[1, 1]))
    logits = tf.add(tf.sparse_tensor_dense_matmul(X, W), b)
    scores = tf.nn.sigmoid(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.eta)
    if cluster_spec is not None:
        num_workers = cluster_spec.num_tasks("worker")
        optimizer = tf.train.SyncReplicasOptimizer(
            opt=optimizer, 
            replicas_to_aggregate=num_workers, 
            total_num_replicas=num_workers, 
            use_locking=True)
    global_step = tf.get_variable(
        name="global_step", shape=[], dtype=tf.int32, 
        initializer=tf.constant_initializer(0), 
        trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    graph = {
        "input": X, 
        "label": y, 
        "output": scores, 
        "loss": loss, 
        "train_op": train_op, 
        "step": global_step, 
        "optimizer": optimizer
    }
    return graph


if __name__ == "__main__":
    define_args()
    FLAGS = tf.app.flags.FLAGS
    assert FLAGS.job_name in ("ps", "worker"), "Unrecognizable job name: " + FLAGS.job_name
    
    train_with_sgd(build_graph)
