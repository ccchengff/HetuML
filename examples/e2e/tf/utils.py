#-*- coding:utf-8 -*-

import json

def parse_config(config_file):
    with open(config_file) as fd:
        cluster = json.load(fd)
    
    assert "ps" in cluster and "worker" in cluster, \
        "config file should contain both 'ps' and 'worker'"

    return cluster


def get_cluster_spec(config_file):
    cluster = parse_config(config_file)
    cluster_spec = tf.train.ClusterSpec(cluster)
    return cluster_spec


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
