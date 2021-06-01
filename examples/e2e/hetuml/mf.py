# -*- coding:utf-8 -*-

from hetuml.decomposition import NMF
from hetuml.data import COOMatrix
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
    parser.add_argument("--embedding_dim", type=int, default=8, help="Size of hidden dimension")
    parser.add_argument("--num_epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--eta", type=float, default=0.1, help="Learning rate")
    
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
        cluster.start_server(NMF.ps_data_type)
    elif args.role == "worker":
        # load data
        train_data = COOMatrix.from_file(args.train_path)
        valid_data = COOMatrix.from_file(args.valid_path)
        
        model = NMF(
            embedding_dim=args.embedding_dim, 
            num_epoch=args.num_epoch, 
            learning_rate=args.eta, 
            parallel=is_distributed)
        
        logging.info("Start training...")
        time_start = time.time()
        model.fit(train_data, valid_data)
        logging.info("Training cost {:.6f} seconds".format(time.time() - time_start))

    if is_distributed:
        cluster.join()
