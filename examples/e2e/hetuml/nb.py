# -*- coding:utf-8 -*-

from hetuml.naive_bayes import NaiveBayes
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
    parser.add_argument("--num_labels", type=int, default=-1, help="Number of labels")
    
    args = parser.parse_args()
    is_distributed = len(args.scheduler) > 0
    if is_distributed:
        cluster = Cluster(
            scheduler=args.scheduler,  
            num_servers=args.num_servers, 
            num_workers=args.num_workers, 
            role=args.role)
    
    if args.role == "server":
        cluster.start_server(NaiveBayes.ps_data_type)
    elif args.role == "worker":
        # load data
        assert args.num_labels > 0, "Please provide number of labels"
        rank = cluster.rank if is_distributed else 0
        num_workers = args.num_workers if is_distributed else 1
        train_data = Dataset.from_file(args.train_path, neg_y=False, 
                                       rank=rank, 
                                       total_ranks=num_workers)
        valid_data = Dataset.from_file(args.valid_path, neg_y=False, 
                                       rank=rank, 
                                       total_ranks=num_workers)
        
        model = NaiveBayes(
            num_label=args.num_labels, 
            metrics="", 
            parallel=is_distributed)
        
        logging.info("Start training...")
        time_start = time.time()
        model.fit(train_data, valid_data)
        logging.info("Training cost {:.6f} seconds".format(time.time() - time_start))

        metrics = model.evaluate(valid_data, ["cross-entropy"])
        logging.info("Validation loss: {:.4f}".format(metrics["cross-entropy"]))

    if is_distributed:
        cluster.join()
