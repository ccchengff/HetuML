# -*- coding:utf-8 -*-

from hetuml.decomposition import LDA
from hetuml.data import Corpus
from hetuml.cluster import Cluster
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
from scipy.sparse import coo_matrix
import numpy as np
import os, sys

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
        cluster.start_server(LDA.ps_data_type)
    elif role == "worker":
        # path to data
        data_path = "./data/pubmed/pubmed.toy"
        num_words = 2845

        corpus = Corpus.from_file(num_words, data_path, 
                                  rank=cluster.rank, 
                                  total_ranks=cluster.num_workers)
        lda = LDA(
            num_words=num_words, 
            num_iters=10, 
            parallel=True)
        test_fit(lda, corpus)
        loaded = test_save_and_load(lda, num_words)
        test_predict(loaded, corpus)
    
    cluster.join()
