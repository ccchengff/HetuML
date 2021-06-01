#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.learn import learn_runner
import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from utils import parse_config

import os
import time
import json
import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def setup_cluster():
    FLAGS = tf.app.flags.FLAGS
    cluster_config = parse_config(FLAGS.config)
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_config, 
        "task": {"type": FLAGS.job_name, "index": FLAGS.task_index}, 
        "environment": "cloud"
        })


def load_data(input_path, rank=0, num_workers=1):
    if input_path.endswith(".npz"):
        if num_workers > 1:
            input_path = input_path[:-4] + "_{}_of_{}.npz".format(rank, num_workers)
        logging.info("Loading from %s..." % input_path)
        loader = np.load(input_path)
        X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                        shape=loader['shape'])
        y = loader['label']
    else:
        logging.info("Loading from %s..." % input_path)
        X, y = load_svmlight_file(input_path, dtype=np.float32)
        if num_workers > 1:
            X = X[rank:][::num_workers]
            y = y[rank:][::num_workers]
    dim = X.shape[1]
    # TensorForest does not support sparse continuous
    X = X.toarray()
    return X, y, dim


def func(output_dir="models"):
    FLAGS = tf.app.flags.FLAGS
    
    if len(FLAGS.config) > 0:
        cluster_config = parse_config(FLAGS.config)
        cluster_spec = tf.train.ClusterSpec(cluster_config)
        server = tf.train.Server(cluster_spec, 
                                job_name=FLAGS.job_name,
                                task_index=FLAGS.task_index)
        is_distributed = True
    else:
        is_distributed = False

    if FLAGS.job_name == "ps":
        with tf.device("/job:ps/task:%d" % FLAGS.task_index):
            done_queue = tf.FIFOQueue(1, tf.int32, shared_name="queue%d" % FLAGS.task_index)
        
        with tf.Session(server.target) as sess:
            sess.run(done_queue.dequeue())
        logging.info("Server terminating")
    else:
        # load datasets
        rank = FLAGS.task_index if is_distributed else 0
        num_workers = len(cluster_config["worker"]) if is_distributed else 1
        train_X, train_y, train_dim = load_data(
            FLAGS.train_path, rank=rank, num_workers=num_workers)
        valid_X, valid_y, valid_dim = load_data(
            FLAGS.valid_path, rank=rank, num_workers=num_workers)
        max_dim = max(train_dim, valid_dim)
        logging.info("Data loading done, #train[{}] #valid[{}] #dim[{}]".format(
            train_y.shape[0], valid_y.shape[0], max_dim))
        
        def _input_fn(X, y):
            features = tf.convert_to_tensor(X)
            labels = tf.convert_to_tensor(y)
            return {"features": features}, labels
        
        def _train_input_fn():
            return _input_fn(train_X, train_y)
        
        def _valid_input_fn():
            return _input_fn(valid_X, valid_y)

        # define estimator
        params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
            num_classes=2, 
            num_features=max_dim, 
            regression=False, 
            num_trees=FLAGS.num_trees, 
            max_nodes=FLAGS.max_nodes, 
            bagging_fraction=FLAGS.ins_sample, 
            feature_bagging_fraction=FLAGS.feat_sample)
        
        if is_distributed:
            dist_strategy = tf.distribute.experimental.ParameterServerStrategy()
        else:
            dist_strategy = None
        run_config = tf.estimator.RunConfig(
            model_dir=output_dir, 
            train_distribute=dist_strategy)

        clf = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(
            params, 
            model_dir=output_dir, 
            config=run_config)
        
        # fit and eval
        logging.info("Start to fit model...")
        fit_start = time.time()
        clf.fit(input_fn=_train_input_fn, max_steps=100000)
        logging.info("Fit cost {:.3f} seconds".format(time.time() - fit_start))

        clf.evaluate(input_fn=_valid_input_fn, steps=1)

        # notify ps
        if is_distributed and FLAGS.task_index == 0:
            ps_done_op = []
            for i in range(cluster_spec.num_tasks("ps")):
                with tf.device("/job:ps/task:%d" % i):
                    done_queue = tf.FIFOQueue(1, tf.int32, shared_name="queue%d" % i)
                    ps_done_op.append(done_queue.enqueue(1))
            with tf.Session(server.target) as sess:
                sess.run(ps_done_op)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string("config", "", "Path to cluster config")
    tf.app.flags.DEFINE_string("job_name", "worker", "Either 'ps' or 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    tf.app.flags.DEFINE_string("train_path", "", "Path to training data")
    tf.app.flags.DEFINE_string("valid_path", "", "Path to validation data")
    tf.app.flags.DEFINE_float("eta", 0.1, "Learning rate")
    tf.app.flags.DEFINE_integer("num_trees", 100, "Number of trees")
    tf.app.flags.DEFINE_integer("max_nodes", 127, "Maximum number of nodes for each tree")
    tf.app.flags.DEFINE_float("ins_sample", 0.3, "Sample ratio for instances")
    tf.app.flags.DEFINE_float("feat_sample", 0.3, "Sample ratio for features")
    FLAGS = tf.app.flags.FLAGS
    assert FLAGS.job_name in ("ps", "worker"), "Unrecognizable job name: " + FLAGS.job_name
    
    if len(FLAGS.config) > 0:
        setup_cluster()
    else:
        assert FLAGS.job_name == "worker", \
            "Job must be worker in standalone mode: " + str(FLAGS.job_name)
        assert FLAGS.task_index == 0, \
            "Index must be 0 in standalone mode: " + str(FLAGS.task_index)

    func()
