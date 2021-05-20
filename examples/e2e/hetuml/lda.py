# -*- coding:utf-8 -*-

from hetuml.decomposition import LDA
from hetuml.data import Corpus
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
    parser.add_argument("--num_words", type=int, default=-1, help="Number of words")
    
    args = parser.parse_args()
    is_distributed = len(args.scheduler) > 0
    if is_distributed:
        cluster = Cluster(
            scheduler=args.scheduler,  
            num_servers=args.num_servers, 
            num_workers=args.num_workers, 
            role=args.role)
    
    if args.role == "server":
        cluster.start_server(LDA.ps_data_type)
    elif args.role == "worker":
        # load data
        rank = cluster.rank if is_distributed else 0
        num_workers = args.num_workers if is_distributed else 1
        train_data = Corpus.from_file(args.num_words, args.train_path, 
                                      rank=rank, 
                                      total_ranks=num_workers)
        valid_data = Corpus.from_file(args.num_words, args.valid_path, 
                                      rank=rank, 
                                      total_ranks=num_workers)
        
        model = LDA(
            num_words=args.num_words, 
            num_topics=100, 
            num_iters=10, 
            parallel=is_distributed)
        
        logging.info("Start training...")
        time_start = time.time()
        model.fit(train_data)
        logging.info("Training cost {:.6f} seconds".format(time.time() - time_start))

        model.predict(valid_data)

    if is_distributed:
        cluster.join()
