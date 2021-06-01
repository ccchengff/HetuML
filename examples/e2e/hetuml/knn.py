# -*- coding:utf-8 -*-

from hetuml.neighbors import KNN
from hetuml.data import Dataset
from hetuml.cluster import Cluster
import numpy as np
from scipy.sparse import csr_matrix

import os, sys
import time
import argparse
import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def load_libsvm(input_path):
    logging.info("Loading data from {}...".format(input_path))
    if input_path.endswith(".npz"):
        loader = np.load(input_path)
        X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                        shape=loader['shape'])
        y = loader['label']
        data = Dataset.from_data((X, y))
    else:
        data = Dataset.from_file(input_path, neg_y=False, 
                                 rank=rank, total_ranks=num_workers)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="", help="Path to training data")
    parser.add_argument("--valid_path", type=str, default="", help="Path to validation data")
    parser.add_argument("--num_labels", type=int, default=-1, help="Number of labels")
    
    args = parser.parse_args()
    
    # load data
    assert args.num_labels > 0, "Please provide number of labels"
    rank, num_workers = 0, 1
    train_data = load_libsvm(args.train_path)
    valid_data = load_libsvm(args.valid_path)
    
    model = KNN(
        num_label=args.num_labels, 
        num_neighbor=5)
    
    logging.info("Start prediction...")
    time_start = time.time()
    metrics = model.evaluate(train_data, valid_data, ["precision"])
    logging.info("Prediction cost {:.6f} seconds".format(time.time() - time_start))
    logging.info("Validation accuracy: {:.4f}".format(metrics["precision"]))
