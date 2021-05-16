# -*- coding:utf-8 -*-

from hetuml.decomposition import LDA
from hetuml.data import Corpus
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
from scipy.sparse import coo_matrix
import numpy as np
import os

LOG_PREFIX = "\u001B[32m ********** "
LOG_SUFFIX = " ********** \u001B[0m"

def test_fit(model, train_data):
    print(LOG_PREFIX + "Test fitting of " + model.name() + "..." + LOG_SUFFIX)
    model.fit(train_data)
    print(LOG_PREFIX + "Test fitting of " + model.name() + " passed" + LOG_SUFFIX)

def test_save_and_load(model, num_words):
    print(LOG_PREFIX + "Test saving & loading of " + model.name() + "..." + LOG_SUFFIX)
    os.system("mkdir -p ./test_models")
    model_path = "./test_models/" + model.name()
    model.save_model(model_path)
    loaded = LDA(num_words=num_words)
    loaded.load_model(model_path)
    print(LOG_PREFIX + "Test saving & loading of " + model.name() + " passed" + LOG_SUFFIX)
    return loaded

def test_predict(model, pred_data):
    print(LOG_PREFIX + "Test prediction of " + model.name() + "..." + LOG_SUFFIX)
    pred = model.predict(pred_data)
    print(LOG_PREFIX + "Test prediction of " + model.name() + " passed" + LOG_SUFFIX)
    return pred

if __name__ == "__main__":
    # path to data
    data_path = "./data/pubmed/pubmed.toy"
    num_words = 2845

    corpus = Corpus.from_file(num_words, data_path)
    lda = LDA(num_words=num_words, num_iters=10)
    test_fit(lda, corpus)
    loaded = test_save_and_load(lda, num_words)
    test_predict(loaded, corpus)
