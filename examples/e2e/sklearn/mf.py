#-*- coding:utf-8 -*-

from scipy.sparse import coo_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

import numpy as np
import time
import sys
import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def load_matrix(path):
    ratings = np.loadtxt(train_path)
    ratings = ratings.transpose()
    row = ratings[0].astype(np.int64)
    col = ratings[1].astype(np.int64)
    data = ratings[2].astype(np.float32)
    return coo_matrix((data, (row, col)))

if __name__ == '__main__':
    try:
        train_path = sys.argv[1]
        valid_path = sys.argv[2]
    except:
        raise Exception("Missing argument: <train_path> <valid_path>")
    
    logging.info("Loading data from {}...".format(train_path))
    train_matrix = load_matrix(train_path)
    logging.info("Loading data from {}...".format(valid_path))
    valid_matrix = load_matrix(valid_path)
    logging.info("Data loading done, train[{}] valid[{}]".format(
        train_matrix.shape, valid_matrix.shape))

    nmf = NMF(
        n_components=8, 
        max_iter=10,  
        verbose=0)
    
    logging.info("Start training...")
    time_start = time.time()
    W = nmf.fit_transform(train_matrix)
    logging.info("Training cost {:.6f} seconds".format(time.time() - time_start))

    H = nmf.components_
    W = nmf.transform(valid_matrix)
    pred = np.dot(W, H)
    pred_data = pred[valid_matrix.row, valid_matrix.col]
    mse = mean_squared_error(valid_matrix.data, pred_data)
    logging.info("Validation mse: {:.4f}".format(mse))
