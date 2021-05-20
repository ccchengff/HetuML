#-*- coding:utf-8 -*-

import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

class LibSVMDataset(object):

    def __init__(self, input_path, batch_size, neg_y=True, rank=0, num_workers=1):
        super(LibSVMDataset, self).__init__()
        if input_path.endswith(".npz"):
            loader = np.load(input_path)
            X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                            shape=loader['shape'])
            y = loader['label']
            if neg_y:
                y[y != 1] = -1
        else:
            X, y = load_svmlight_file(input_path, dtype=np.float32)
        if num_workers > 1:
            X = X[rank:][::num_workers]
            y = y[rank:][::num_workers]
            if batch_size > 0:
                batch_size = (batch_size + num_workers - 1) // num_workers
        self.features = X
        self.labels = y.reshape(-1, 1)
        self.num_ins, self.max_dim = self.features.shape
        self.batch_size = batch_size if batch_size > 0 else self.num_ins
        self.num_batch = (self.num_ins + batch_size - 1) // batch_size
        self.cursor = 0
    
    def next_batch(self):
        batch_id = self.cursor
        self.cursor += 1
        if self.cursor == self.num_batch:
            self.cursor = 0
        return self.__getitem__(batch_id)
    
    def expand_dim(self, max_dim):
        assert max_dim >= self.max_dim
        self.max_dim = max_dim

    def __len__(self):
        return self.num_batch

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.features.shape[0])
        batch_feats = self.features[start:end].tocoo()
        indices = np.dstack([batch_feats.row, batch_feats.col])[0]
        values = batch_feats.data
        batch_feats = (indices, values, (end - start, self.max_dim))
        batch_labels = self.labels[start:end]
        return batch_feats, batch_labels
