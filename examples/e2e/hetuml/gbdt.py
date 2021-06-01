# -*- coding:utf-8 -*-

from hetuml.ensemble import GBDT
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

def load_libsvm(input_path, rank, num_workers):
    if input_path.endswith(".npz"):
        if num_workers > 1:
            input_path = input_path[:-4] + "_{}_of_{}.npz".format(rank, num_workers)
        logging.info("Loading data from {}...".format(input_path))
        loader = np.load(input_path)
        X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                        shape=loader['shape'])
        y = loader['label']
        data = Dataset.from_data((X, y))
    else:
        logging.info("Loading data from {}...".format(input_path))
        data = Dataset.from_file(input_path, neg_y=False, 
                                 rank=rank, total_ranks=num_workers)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, default="worker", help="Role", choices=("scheduler", "server", "worker"))
    parser.add_argument("--scheduler", type=str, default="", help="Address of scheduler")
    parser.add_argument("--num_servers", type=int, default=1, help="Number of parameter servers")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--train_path", type=str, default="", help="Path to training data")
    parser.add_argument("--valid_path", type=str, default="", help="Path to validation data")
    parser.add_argument("--eta", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--num_trees", type=int, default=10, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum tree depths")

    args = parser.parse_args()
    is_distributed = len(args.scheduler) > 0
    if is_distributed:
        logging.info("Setting up cluster...")
        cluster = Cluster(
            scheduler=args.scheduler,  
            num_servers=args.num_servers, 
            num_workers=args.num_workers, 
            role=args.role)
        logging.info("Set up cluster successfully")
    
    if args.role == "server":
        cluster.start_server(GBDT.ps_data_type)
    elif args.role == "worker":
        # load data
        rank = cluster.rank if is_distributed else 0
        num_workers = args.num_workers if is_distributed else 1
        train_data = load_libsvm(args.train_path, 
                                 rank=rank, 
                                 num_workers=num_workers)
        valid_data = load_libsvm(args.valid_path, 
                                 rank=rank, 
                                 num_workers=num_workers)
        
        model = GBDT(
            num_round=args.num_trees, 
            learning_rate=args.eta, 
            num_split=20, 
            max_depth=args.max_depth, 
            metrics="log-loss,error,precision", 
            parallel=is_distributed)
        
        logging.info("Start training...")
        time_start = time.time()
        model.fit(train_data, valid_data)
        logging.info("Training cost {:.6f} seconds".format(time.time() - time_start))

    if is_distributed:
        cluster.join()
