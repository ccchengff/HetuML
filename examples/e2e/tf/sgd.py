#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
from libsvm_dataset import LibSVMDataset
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
    tf.app.flags.DEFINE_float("eta", 0.1, "Learning rate")
    tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size")
    tf.app.flags.DEFINE_integer("print_freq", 100, "Frequency to print")


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

        logging.info("Loading training data from '{}'...".format(FLAGS.train_path))
        train_data = LibSVMDataset(FLAGS.train_path, FLAGS.batch_size, 
                                   rank=FLAGS.task_index, 
                                   num_workers=cluster_spec.num_tasks("worker") if is_distributed else 1)
        logging.info("Loading done, #ins[{}] #dim[{}]"
                     .format(train_data.num_ins, train_data.max_dim))
        if len(FLAGS.valid_path) > 0:
            logging.info("Loading validation data from '{}'...".format(FLAGS.valid_path))
            valid_data = LibSVMDataset(FLAGS.valid_path, FLAGS.batch_size, 
                                       rank=FLAGS.task_index, 
                                       num_workers=cluster_spec.num_tasks("worker") if is_distributed else 1)
            logging.info("Loading done, #ins[{}] #dim[{}]"
                        .format(valid_data.num_ins, valid_data.max_dim))
            if train_data.max_dim > valid_data.max_dim:
                valid_data.expand_dim(train_data.max_dim)
            elif valid_data.max_dim > train_data.max_dim:
                train_data.expand_dim(valid_data.max_dim)

        if is_distributed:
            device = tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster_spec)
        else:
            device = "/cpu:0"
        with tf.device(device):
            graph = build_graph_fn(train_data.max_dim, cluster_spec)
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
            batch_time = AverageMeter()
            batch_loss = AverageMeter()
            train_start = time.time()

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
                    logging.info("Step[{}] Elapsed[{:.4f} sec] "
                                "Time[{batch.val:.4f}] ({batch.avg:.4f}) "
                                "Loss {loss.val:.4f} ({loss.avg:.4f})"
                                .format(step_val, batch_end - train_start, 
                                        batch=batch_time, loss=batch_loss))
            logging.info("Training cost {:.3f} seconds, loss = {:.4f}".format(
                time.time() - train_start, batch_loss.avg))
            
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
