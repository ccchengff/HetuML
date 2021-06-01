#-*- coding:utf-8 -*-

from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
import numpy as np

import time
import sys
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
    else:
        X, y = load_svmlight_file(input_path, dtype=np.float32)
    return X, y

if __name__ == '__main__':
    try:
        train_path = sys.argv[1]
        valid_path = sys.argv[2]
    except:
        raise Exception("Missing argument: <train_path> <valid_path>")
    
    train_X, train_y = load_libsvm(train_path)
    valid_X, valid_y = load_libsvm(valid_path)
    max_dim = max(train_X.shape[1], valid_X.shape[1])
    if train_X.shape[1] != max_dim:
        train_X.resize(train_X.shape[0], max_dim)
    if valid_X.shape[1] != max_dim:
        valid_X.resize(valid_X.shape[0], max_dim)
    train_X = train_X.toarray()
    valid_X = valid_X.toarray()
    logging.info("Data loading done, #train[{}] #valid[{}] #dim[{}]".format(
        train_y.shape[0], valid_y.shape[0], max_dim))

    clf = GaussianNB()
    
    logging.info("Start training...")
    time_start = time.time()
    clf.fit(train_X, train_y)
    logging.info("Training cost {:.6f} seconds".format(time.time() - time_start))

    acc = (pred.argmax(axis=1) == valid_y).sum() / valid_y.shape[0]
    logging.info("Validation accuracy: {:.4f}".format(acc))
