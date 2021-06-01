#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from utils import parse_config, AverageMeter

import time
import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def define_args():
    tf.app.flags.DEFINE_string("config", "", "Path to cluster config")
    tf.app.flags.DEFINE_string("job_name", "worker", "Either 'ps' or 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    tf.app.flags.DEFINE_string("train_path", "", "Path to training data")
    tf.app.flags.DEFINE_string("valid_path", "", "Path to validation data")
    tf.app.flags.DEFINE_integer("num_epoch", 1, "Number of epochs")
    tf.app.flags.DEFINE_float("eta", 0.1, "Learning rate")
    tf.app.flags.DEFINE_integer("batch_size", 10000, "Batch size")
    tf.app.flags.DEFINE_integer("print_freq", 10, "Frequency to print")


def train_with_sgd(build_graph_fn):
    FLAGS = tf.app.flags.FLAGS
    if len(FLAGS.config) == 0:
        assert FLAGS.job_name == "worker", \
            "Job must be worker in standalone mode: " + str(FLAGS.job_name)
        assert FLAGS.task_index == 0, \
            "Index must be 0 in standalone mode: " + str(FLAGS.task_index)
        cluster_spec = None
    else:
        cluster_config = parse_config(FLAGS.config)
        cluster_spec = tf.train.ClusterSpec(cluster_config)
        server = tf.train.Server(cluster_spec, 
                                job_name=FLAGS.job_name,
                                task_index=FLAGS.task_index)
    
    is_distributed = (cluster_spec is not None)

    if FLAGS.job_name == "ps":
        with tf.device("/job:ps/task:%d" % FLAGS.task_index):
            done_queue = tf.FIFOQueue(1, tf.int32, shared_name="queue%d" % FLAGS.task_index)
        
        with tf.Session(server.target) as sess:
            sess.run(done_queue.dequeue())
        logging.info("Server terminating")
    else:
        if is_distributed and FLAGS.task_index == 0:
            ps_done_op = []
            for i in range(cluster_spec.num_tasks("ps")):
                with tf.device("/job:ps/task:%d" % i):
                    done_queue = tf.FIFOQueue(1, tf.int32, shared_name="queue%d" % i)
                    ps_done_op.append(done_queue.enqueue(1))

        train_data = LibSVMDataset(FLAGS.train_path, FLAGS.batch_size, 
                                   rank=FLAGS.task_index, 
                                   num_workers=cluster_spec.num_tasks("worker") if is_distributed else 1)
        logging.info("Loading training data done, #ins[{}] #dim[{}]"
                     .format(train_data.num_ins, train_data.max_dim))
        max_dim = train_data.max_dim
        if len(FLAGS.valid_path) > 0:
            valid_data = LibSVMDataset(FLAGS.valid_path, FLAGS.batch_size, 
                                       rank=FLAGS.task_index, 
                                       num_workers=cluster_spec.num_tasks("worker") if is_distributed else 1)
            logging.info("Loading validation data done, #ins[{}] #dim[{}]"
                        .format(valid_data.num_ins, valid_data.max_dim))
            max_dim = max(train_data.max_dim, valid_data.max_dim)
            if valid_data.max_dim < max_dim:
                valid_data.expand_dim(max_dim)
            elif train.max_dim < max_dim:
                train_data.expand_dim(max_dim)

        if is_distributed:
            device = tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster_spec)
        else:
            device = "/cpu:0"
        with tf.device(device):
            graph = build_graph_fn(max_dim, cluster_spec)
            hooks = []
            if is_distributed:
                sync_replicas_hook = graph["optimizer"].make_session_run_hook(
                    (FLAGS.task_index == 0), num_tokens=0)
                hooks.append(sync_replicas_hook)
            init_op = tf.global_variables_initializer()
            
            mon_sess = tf.train.MonitoredTrainingSession(
                master=server.target if is_distributed else None, 
                is_chief=(FLAGS.task_index == 0), 
                hooks=hooks, 
                config=tf.ConfigProto(
                    allow_soft_placement=True, 
                    log_device_placement=False)
                )
            if FLAGS.task_index == 0:
                mon_sess.run(init_op)
            
            logging.info("Start training")
            train_start = time.time()
            for epoch in range(FLAGS.num_epoch):
                epoch_start = time.time()
                batch_time = AverageMeter()
                batch_loss = AverageMeter()
                for batch_id in range(train_data.num_batch):
                    batch_start = time.time()
                    batch_x, batch_y = train_data.next_batch()
                    x, y = graph["input"], graph["label"]
                    loss, train_op, step = \
                        graph["loss"], graph["train_op"], graph["step"]
                    _, loss_val, step_val = mon_sess.run(
                        [train_op, loss, step], 
                        feed_dict={x:batch_x, y:batch_y})
                    
                    batch_end = time.time()
                    batch_time.update(time.time() - batch_start)
                    batch_loss.update(loss_val)
                    if (batch_id + 1) % FLAGS.print_freq == 0:
                        logging.info("Epoch[{}] Step[{}] "
                                    "Time[{batch.val:.4f}] ({batch.avg:.4f}) "
                                    "Loss {loss.val:.4f} ({loss.avg:.4f})"
                                    .format(epoch, step_val, 
                                            batch=batch_time, loss=batch_loss))
                logging.info("Epoch[{}] cost {:.3f} seconds, avgerage loss = {:.4f}".format(
                    epoch, time.time() - epoch_start, batch_loss.avg))
            logging.info("Training cost {:.3f} seconds".format(time.time() - train_start))
            
            if len(FLAGS.valid_path) > 0:
                logging.info("Evaluation on valid data")
                valid_loss = AverageMeter()
                for _ in range(valid_data.num_batch):
                    batch_x, batch_y = valid_data.next_batch()
                    x, y = graph["input"], graph["label"]
                    loss_val, = mon_sess.run(
                        [graph["loss"]], 
                        feed_dict={x:batch_x, y:batch_y})
                    valid_loss.update(loss_val)
                logging.info("Validation loss = {:.4f}".format(valid_loss.avg))
                        
            # notify ps
            if is_distributed and FLAGS.task_index == 0:
                with tf.Session(server.target) as sess:
                    sess.run(ps_done_op)
            
            logging.info("Worker terminating")


class LibSVMDataset(object):

    def __init__(self, input_path, batch_size, rank=0, num_workers=1):
        super(LibSVMDataset, self).__init__()
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

        self.features = X
        self.labels = y.reshape(-1, 1)
        self.num_ins, self.max_dim = self.features.shape
        self.batch_size = batch_size if batch_size > 0 else self.num_ins
        self.num_batch = (self.num_ins + batch_size - 1) // batch_size
        self.cursor = 0
    
    def next_batch(self):
        batch_id = self.cursor
        self.cursor += 1
        if self.cursor == self.num_batch:
            self.cursor = 0
        return self.__getitem__(batch_id)
    
    def expand_dim(self, max_dim):
        assert max_dim >= self.max_dim
        self.max_dim = max_dim

    def __len__(self):
        return self.num_batch

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.features.shape[0])
        batch_feats = self.features[start:end].tocoo()
        indices = np.dstack([batch_feats.row, batch_feats.col])[0]
        values = batch_feats.data
        batch_feats = (indices, values, (end - start, self.max_dim))
        batch_labels = self.labels[start:end]
        return batch_feats, batch_labels
