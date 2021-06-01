#-*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import numpy as np
import time
import sys
import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def load(file_name):
    corpus = []
    with open(file_name) as fd:
        for line in fd:
            # skip docId
            start = line.find('\t') + 1
            corpus.append(line[start:])
    return corpus

if __name__ == '__main__':
    try:
        train_path = sys.argv[1]
        valid_path = sys.argv[2]
        num_epoch = 10
    except:
        raise Exception("Missing argument: <train_path> <valid_path>")
    
    logging.info("Loading data from {}...".format(train_path))
    train_corpus = load(train_path)
    vectorizer = CountVectorizer()
    train_corpus = vectorizer.fit_transform(train_corpus)
    logging.info("Loading data from {}...".format(valid_path))
    valid_corpus = load(valid_path)
    valid_corpus = vectorizer.transform(valid_corpus)
    logging.info("Data loading done, train[{}] valid[{}]".format(
        train_corpus.shape, valid_corpus.shape))
    
    lda = LatentDirichletAllocation(
        n_components=100, # number of topics
        doc_topic_prior=0.5, 
        topic_word_prior=0.01,
        max_iter=1,
        learning_method='batch',
        total_samples=train_corpus.shape[0],
        batch_size=train_corpus.shape[0],
        evaluate_every=10000,
        verbose=1)
    
    logging.info("Start training...")
    time_start = time.time()
    for i in range(num_epoch):
        t0 = time.time()
        lda.partial_fit(train_corpus)
        llh = lda.score(valid_corpus)
        t1 = time.time()
        logging.info("Iter[{}] cost {:.6f} seconds ({:.6f} seconds elapsed), loglikelihood: {:.6f}".format(
            i, t1 - t0, t1 - time_start, llh))
    logging.info("Training cost {:.6f} seconds".format(time.time() - time_start))
