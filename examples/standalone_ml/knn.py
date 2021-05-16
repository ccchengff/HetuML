# -*- coding:utf-8 -*-

from hetuml.neighbors import KNN
from hetuml.data import Dataset
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
import os

LOG_PREFIX = "\u001B[32m ********** "
LOG_SUFFIX = " ********** \u001B[0m"

def test_predict(model, labeled_data, pred_data, labels=None):
    print(LOG_PREFIX + "Test prediction of " + model.name() + "..." + LOG_SUFFIX)
    pred = model.predict_proba(labeled_data, pred_data)
    if labels is not None:
        loss = log_loss(labels, pred)
        print("Log loss: {:.6f}".format(loss))
    print(LOG_PREFIX + "Test prediction of " + model.name() + " passed" + LOG_SUFFIX)
    return pred

def test_evaluate(model, labeled_data, eval_data):
    print(LOG_PREFIX + "Test evaluation of " + model.name() + "..." + LOG_SUFFIX)
    metrics = model.evaluate(labeled_data, eval_data, ["cross-entropy", "error", "precision"])
    print("eval metrics: {}".format(metrics))
    print(LOG_PREFIX + "Test evaluation of " + model.name() + " passed" + LOG_SUFFIX)


if __name__ == "__main__":
    # path to data
    train_path = "./data/satimage/satimage.scale"
    valid_path = "./data/satimage/satimage.scale.t"

    # option 1: directly read from file
    train_data = Dataset.from_file(train_path)
    valid_data = Dataset.from_file(train_path)
    knn = KNN(num_label=6, num_neighbor=5)
    test_predict(knn, train_data, valid_data)
    test_evaluate(knn, train_data, valid_data)

    # option 2: load as csr_matrix
    X_train, y_train = load_svmlight_file(train_path)
    train_data = (X_train, y_train)
    X_valid, y_valid = load_svmlight_file(valid_path)
    valid_data = (X_valid, y_valid)
    knn = KNN(num_label=6, num_neighbor=5)
    test_predict(knn, train_data, X_valid, labels=y_valid)
    test_evaluate(knn, train_data, valid_data)
