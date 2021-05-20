#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.learn import learn_runner
from libsvm_dataset import LibSVMDataset
from utils import parse_config
import numpy as np

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
        num_workers = len(cluster_config["worker"]) if is_distributed else 1
        train_data = LibSVMDataset(FLAGS.train_path, -1, 
                                   rank=FLAGS.task_index, 
                                   num_workers=num_workers)
        valid_data = LibSVMDataset(FLAGS.valid_path, -1, 
                                   rank=FLAGS.task_index, 
                                   num_workers=num_workers)
        max_dim = max(train_data.max_dim, valid_data.max_dim)

        def to_coo(dataset):
            X, y = dataset.features.tocoo(), dataset.labels
            indices = np.dstack([X.row, X.col])[0]
            values = X.data
            X = (indices, values)
            return X, y
        
        train_X, train_y = to_coo(train_data)
        valid_X, valid_y = to_coo(valid_data)
        logging.info("Data loading done, #train[{}] #valid[{}] #dim[{}]".format(
            train_y.shape[0], valid_y.shape[0], max_dim))
        
        def _input_fn(X, y):
            features = tf.SparseTensor(indices=X[0], values=X[1], 
                                       dense_shape=(y.shape[0], max_dim))
            labels = tf.convert_to_tensor(y)
            return {"features": features}, labels
        
        def _train_input_fn():
            return _input_fn(train_X, train_y)
        
        def _valid_input_fn():
            return _input_fn(valid_X, valid_y)

        # define estimator
        learner_config = learner_pb2.LearnerConfig()

        examples_per_layer = train_y.shape[0]
        learner_config.learning_rate_tuner.fixed.learning_rate = FLAGS.eta
        learner_config.regularization.l1 = 0.0
        learner_config.regularization.l2 = 1.0 / examples_per_layer
        learner_config.constraints.max_tree_depth = FLAGS.max_depth
        learner_config.growing_mode = learner_pb2.LearnerConfig.WHOLE_TREE

        if is_distributed:
            dist_strategy = tf.distribute.experimental.ParameterServerStrategy()
        else:
            dist_strategy = None
        run_config = tf.estimator.RunConfig(
            model_dir=output_dir, 
            train_distribute=dist_strategy)

        clf = GradientBoostedDecisionTreeClassifier(
            learner_config=learner_config,
            examples_per_layer=examples_per_layer,
            model_dir=output_dir,
            num_trees=FLAGS.num_trees,
            center_bias=False,
            config=run_config,
            num_quantiles=20)
        
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
    tf.app.flags.DEFINE_integer("num_trees", 10, "Number of trees")
    tf.app.flags.DEFINE_integer("max_depth", 4, "Maximum depth for each tree")
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
