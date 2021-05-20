#-*- coding:utf-8 -*-

from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss

import time
import sys
import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

if __name__ == '__main__':
    try:
        train_path = sys.argv[1]
        valid_path = sys.argv[2]
    except:
        raise Exception("Missing argument: <train_path> <valid_path>")
    
    logging.info("Loading data from {}...".format(train_path))
    train_X, train_y = load_svmlight_file(train_path)
    logging.info("Loading data from {}...".format(valid_path))
    valid_X, valid_y = load_svmlight_file(valid_path)
    max_dim = max(train_X.shape[1], valid_X.shape[1])
    if train_X.shape[1] != max_dim:
        train_X.resize(train_X.shape[0], max_dim)
    if valid_X.shape[1] != max_dim:
        valid_X.resize(valid_X.shape[0], max_dim)
    logging.info("Data loading done, #train[{}] #valid[{}] #dim[{}]".format(
        train_y.shape[0], valid_y.shape[0], max_dim))

    clf = KNeighborsClassifier(
        n_neighbors=5, 
        algorithm='brute')
    
    logging.info("Start prediction...")
    time_start = time.time()
    clf.fit(train_X, train_y)
    pred = clf.predict_proba(valid_X)
    time_end = time.time()
    logging.info("Prediction cost {:.6f} seconds".format(time.time() - time_start))

    loss = log_loss(valid_y, pred)
    logging.info("Validation loss: {:.4f}".format(loss))

    acc = (pred.argmax(axis=1) == valid_y).sum() / valid_y.shape[0]
    logging.info("Validation accuracy: {:.4f}".format(acc))
