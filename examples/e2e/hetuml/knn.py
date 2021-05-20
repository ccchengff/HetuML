# -*- coding:utf-8 -*-

from hetuml.neighbors import KNN
from hetuml.data import Dataset
from hetuml.cluster import Cluster

import os, sys
import time
import argparse
import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="", help="Path to training data")
    parser.add_argument("--valid_path", type=str, default="", help="Path to validation data")
    parser.add_argument("--num_labels", type=int, default=-1, help="Number of labels")
    
    args = parser.parse_args()
    
    # load data
    assert args.num_labels > 0, "Please provide number of labels"
    rank, num_workers = 0, 1
    train_data = Dataset.from_file(args.train_path, neg_y=False, 
                                    rank=rank, 
                                    total_ranks=num_workers)
    valid_data = Dataset.from_file(args.valid_path, neg_y=False, 
                                    rank=rank, 
                                    total_ranks=num_workers)
    
    model = KNN(
        num_label=args.num_labels, 
        num_neighbor=5)
    
    logging.info("Start prediction...")
    time_start = time.time()
    metrics = model.evaluate(train_data, valid_data, ["precision"])
    logging.info("Prediction cost {:.6f} seconds".format(time.time() - time_start))
    logging.info("Validation accuracy: {:.4f}".format(metrics["precision"]))
