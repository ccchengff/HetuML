# -*- coding:utf-8 -*-

from hetuml.linear import LogisticRegression
from hetuml.data import Dataset
from hetuml.cluster import Cluster
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
import os, sys

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

def test_predict(model, pred_data, labels=None):
    print(LOG_PREFIX + "Test prediction of " + model.name() + "..." + LOG_SUFFIX)
    pred = model.predict_proba(pred_data)
    if labels is not None:
        labels_cp = labels.copy()
        labels_cp[labels_cp == -1] = 0
        loss = log_loss(labels_cp, pred)
        print("Log loss: {:.6f}".format(loss))
    print(LOG_PREFIX + "Test prediction of " + model.name() + " passed" + LOG_SUFFIX)
    return pred

def test_evaluate(model, eval_data):
    print(LOG_PREFIX + "Test evaluation of " + model.name() + "..." + LOG_SUFFIX)
    metrics = model.evaluate(eval_data, ["log-loss", "error", "precision"])
    print("eval metrics: {}".format(metrics))
    print(LOG_PREFIX + "Test evaluation of " + model.name() + " passed" + LOG_SUFFIX)

if __name__ == "__main__":
    try:
        role = sys.argv[1]
    except:
        print("Missing argument: <role>")
        exit(1)

    cluster = Cluster(
        scheduler="127.0.0.1:50021", 
        num_servers=1, 
        num_workers=1, 
        role=role)
    
    if role == "server":
        cluster.start_server(LogisticRegression.ps_data_type)
    elif role == "worker":
        # path to data
        train_path = "./data/a9a/a9a"
        valid_path = "./data/a9a/a9a.t"

        # option 1: directly read from file
        train_data = Dataset.from_file(train_path, neg_y=True, 
                                       rank=cluster.rank, 
                                       total_ranks=cluster.num_workers)
        valid_data = Dataset.from_file(train_path, neg_y=True, 
                                       rank=cluster.rank, 
                                       total_ranks=cluster.num_workers)
        lr = LogisticRegression(
            learning_rate=0.5, 
            metrics="log-loss,error,precision", 
            parallel=True)
        test_fit(lr, train_data, valid_data)
        loaded = test_save_and_load(lr)
        test_predict(loaded, valid_data)
        test_evaluate(loaded, valid_data)

        # option 2: load as csr_matrix and slice by rank
        def load_and_slice(path, neg_y):
            X, y = load_svmlight_file(path)
            if cluster.num_workers > 1:
                X = X[cluster.rank:][::cluster.num_workers]
                y = y[cluster.rank:][::cluster.num_workers]
            if neg_y:
                y[y != 1] = -1
            return X, y

        X_train, y_train = load_and_slice(train_path, neg_y=True)
        train_data = (X_train, y_train)
        X_valid, y_valid = load_and_slice(valid_path, neg_y=True)
        valid_data = (X_valid, y_valid)
        lr = LogisticRegression(
            learning_rate=0.5, 
            metrics="log-loss,error,precision", 
            parallel=True)
        test_fit(lr, train_data, valid_data)
        loaded = test_save_and_load(lr)
        test_predict(loaded, X_valid, labels=y_valid)
        test_evaluate(loaded, valid_data)

    cluster.join()
