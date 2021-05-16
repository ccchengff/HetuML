# -*- coding:utf-8 -*-

from hetuml.ensemble import GBDT
from hetuml.data import Dataset
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
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

def test_predict(model, pred_data, labels=None):
    print(LOG_PREFIX + "Test prediction of " + model.name() + "..." + LOG_SUFFIX)
    pred = model.predict_proba(pred_data)
    if labels is not None:
        loss = log_loss(labels, pred)
        print("Log loss: {:.6f}".format(loss))
    print(LOG_PREFIX + "Test prediction of " + model.name() + " passed" + LOG_SUFFIX)
    return pred

def test_evaluate(model, eval_data):
    print(LOG_PREFIX + "Test evaluation of " + model.name() + "..." + LOG_SUFFIX)
    metrics = model.evaluate(eval_data, ["log-loss", "error", "precision"])
    print("eval metrics: {}".format(metrics))
    print(LOG_PREFIX + "Test evaluation of " + model.name() + " passed" + LOG_SUFFIX)

if __name__ == "__main__":
    # path to data
    train_path = "./data/a9a/a9a"
    valid_path = "./data/a9a/a9a.t"

    # option 1: directly read from file
    train_data = Dataset.from_file(train_path)
    valid_data = Dataset.from_file(train_path)
    gbdt = GBDT(max_depth=6, metrics="log-loss,error,precision")
    test_fit(gbdt, train_data, valid_data)
    loaded = test_save_and_load(gbdt)
    test_predict(loaded, valid_data)
    test_evaluate(loaded, valid_data)

    # option 2: load as csr_matrix
    X_train, y_train = load_svmlight_file(train_path)
    train_data = (X_train, y_train)
    X_valid, y_valid = load_svmlight_file(valid_path)
    valid_data = (X_valid, y_valid)
    gbdt = GBDT(max_depth=6, metrics="log-loss,error,precision")
    test_fit(gbdt, train_data, valid_data)
    loaded = test_save_and_load(gbdt)
    test_predict(loaded, X_valid, labels=y_valid)
    test_evaluate(loaded, valid_data)
