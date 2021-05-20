# -*- coding:utf-8 -*-

from hetuml.ensemble import RanomBoostedForest
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
    parser.add_argument("--role", type=str, default="worker", help="Role", choices=("scheduler", "server", "worker"))
    parser.add_argument("--scheduler", type=str, default="", help="Address of scheduler")
    parser.add_argument("--num_servers", type=int, default=1, help="Number of parameter servers")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--train_path", type=str, default="", help="Path to training data")
    parser.add_argument("--valid_path", type=str, default="", help="Path to validation data")
    parser.add_argument("--eta", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--num_trees", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depths")

    args = parser.parse_args()
    is_distributed = len(args.scheduler) > 0
    if is_distributed:
        cluster = Cluster(
            scheduler=args.scheduler,  
            num_servers=args.num_servers, 
            num_workers=args.num_workers, 
            role=args.role)
    
    if args.role == "server":
        cluster.start_server(RanomBoostedForest.ps_data_type)
    elif args.role == "worker":
        # load data
        rank = cluster.rank if is_distributed else 0
        num_workers = args.num_workers if is_distributed else 1
        train_data = Dataset.from_file(args.train_path, neg_y=False, 
                                       rank=rank, 
                                       total_ranks=num_workers)
        valid_data = Dataset.from_file(args.valid_path, neg_y=False, 
                                       rank=rank, 
                                       total_ranks=num_workers)
        
        model = RanomBoostedForest(
            num_round=args.num_trees, 
            learning_rate=args.eta, 
            num_split=20, 
            max_depth=args.max_depth, 
            ins_sp_ratio=0.3, 
            feat_sp_ratio=0.3, 
            metrics="log-loss,error,precision", 
            parallel=is_distributed)
        
        logging.info("Start training...")
        time_start = time.time()
        model.fit(train_data, valid_data)
        logging.info("Training cost {:.6f} seconds".format(time.time() - time_start))

    if is_distributed:
        cluster.join()
