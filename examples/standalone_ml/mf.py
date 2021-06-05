# -*- coding:utf-8 -*-

from hetuml.decomposition import NMF
from hetuml.data import COOMatrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
from scipy.sparse import coo_matrix
import numpy as np
import os

LOG_PREFIX = "\u001B[32m ********** "
LOG_SUFFIX = " ********** \u001B[0m"

def test_fit(model, train_data, valid_data):
    print(LOG_PREFIX + "Test fitting of " + model.name() + "..." + LOG_SUFFIX)
    model.fit(train_data, valid_data)
    print(LOG_PREFIX + "Test fitting of " + model.name() + " passed" + LOG_SUFFIX)

def test_save_and_load(model):
    print(LOG_PREFIX + "Test saving & loading of " + model.name() + "..." + LOG_SUFFIX)
    os.system("mkdir -p ./test_models")
    model_path = "./test_models/" + model.name()
    model.save_model(model_path)
    loaded = type(model)()
    loaded.load_model(model_path)
    print(LOG_PREFIX + "Test saving & loading of " + model.name() + " passed" + LOG_SUFFIX)
    return loaded

def test_predict(model, pred_data):
    print(LOG_PREFIX + "Test prediction of " + model.name() + "..." + LOG_SUFFIX)
    pred = model.predict(pred_data)
    if isinstance(pred_data, coo_matrix):
        square_errors = np.power((pred - pred_data.data), 2)
        mse = square_errors.sum() / square_errors.shape[0]
        print("MSE: {:.6f}".format(mse))
    print(LOG_PREFIX + "Test prediction of " + model.name() + " passed" + LOG_SUFFIX)
    return pred

if __name__ == "__main__":
    # path to data
    train_path = "./data/movielens/ml.toy.tr"
    valid_path = "./data/movielens/ml.toy.te"

    # option 1: directly read from file
    train_data = COOMatrix.from_file(train_path)
    valid_data = COOMatrix.from_file(train_path)
    nmf = NMF()
    test_fit(nmf, train_data, valid_data)
    loaded = test_save_and_load(nmf)
    test_predict(loaded, valid_data)

    # option 2: load as csr_matrix
    def load_matrix(path):
        ratings = np.loadtxt(train_path)
        ratings = ratings.transpose()
        row = ratings[0].astype(np.int64)
        col = ratings[1].astype(np.int64)
        data = ratings[2].astype(np.float32)
        return coo_matrix((data, (row, col)))
    
    train_data = load_matrix(train_path)
    valid_data = load_matrix(valid_path)
    nmf = NMF()
    test_fit(nmf, train_data, valid_data)
    loaded = test_save_and_load(nmf)
    test_predict(loaded, valid_data)
