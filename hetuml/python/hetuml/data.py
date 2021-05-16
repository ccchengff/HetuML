# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from hetuml import _C

from typing import Union, Tuple

class CSRMatrix(object):
    __create_key = object()

    @staticmethod
    def from_data(X: Union[csr_matrix, csc_matrix, coo_matrix]):
        if isinstance(X, (csc_matrix, coo_matrix)):
            X = X.tocsr()
        elif not isinstance(X, csr_matrix):
            raise TypeError("Type {} cannot be converted to csr matrix"
                            .format(type(X)))
        handle = _C.DataMatrixWrapper(X.indices, X.indptr, X.data)
        return CSRMatrix(CSRMatrix.__create_key, handle)

    def __init__(self, create_key, handle: _C.DataMatrixWrapper):
        super(CSRMatrix, self).__init__()
        assert(create_key == CSRMatrix.__create_key), \
            "CSRMatrix objects must be created by CSRMatrix.from_data"
        self._handle = handle
    
    def get_num_rows(self) -> int:
        return self._handle.get_num_instances()
    
    def get_num_cols(self) -> int:
        return self._handle.get_max_dim()


class COOMatrix(object):
    __create_key = object()

    @staticmethod
    def from_file(path: str):
        handle = _C.COOMatrixWrapper(path)
        return COOMatrix(COOMatrix.__create_key, handle)

    @staticmethod
    def from_data(X: Union[csr_matrix, csc_matrix, coo_matrix]):
        if isinstance(X, (csr_matrix, csc_matrix)):
            X = X.tocoo()
        elif not isinstance(X, coo_matrix):
            raise TypeError("Type {} cannot be converted to coo matrix"
                            .format(type(X)))
        handle = _C.COOMatrixWrapper(X.row, X.col, X.data)
        return COOMatrix(COOMatrix.__create_key, handle)

    def __init__(self, create_key, handle: _C.COOMatrixWrapper):
        super(COOMatrix, self).__init__()
        assert(create_key == COOMatrix.__create_key), \
            "COOMatrix objects must be created by "\
            "COOMatrix.from_file or COOMatrix.from_data"
        if not isinstance(handle, _C.COOMatrixWrapper):
            raise TypeError("Invalid type: {}".format(type(handle)))
        self._handle = handle
    
    def get_num_rows(self) -> int:
        return self._handle.get_num_rows()
    
    def get_num_cols(self) -> int:
        return self._handle.get_num_cols()


class Dataset(object):
    __create_key = object()

    @staticmethod
    def from_file(path: str, 
                  data_format: str = "libsvm", 
                  neg_y: bool = False, 
                  rank: int = 0, 
                  total_ranks: int = 1):
        handle = _C.DatasetWrapper(path, data_format, neg_y, rank, total_ranks)
        return Dataset(Dataset.__create_key, handle)
    
    @staticmethod
    def from_data(data: Tuple[Union[csr_matrix, csc_matrix, coo_matrix], np.array]):
        if not isinstance(data, (tuple, list)):
            raise TypeError("Expected a tuple for X and y, got {}"
                            .format(type(data)))
        if len(data) != 2:
            raise TypeError("Expected a tuple for X and y, got {} elements"
                            .format(len(data)))
        X, y = data
        if isinstance(X, (csc_matrix, coo_matrix)):
            X = X.tocsr()
        elif not isinstance(X, csr_matrix):
            raise TypeError("Type {} cannot be converted to csr matrix"
                            .format(type(X)))
        if not isinstance(y, np.ndarray):
            raise TypeError("y should be numpy array, got {}".format(type(y)))
        handle = _C.DatasetWrapper(y, X.indices, X.indptr, X.data)
        return Dataset(Dataset.__create_key, handle)

    def __init__(self, create_key, handle: _C.DatasetWrapper):
        super(Dataset, self).__init__()
        assert(create_key == Dataset.__create_key), \
            "Dataset objects must be created by "\
            "Dataset.from_file or Dataset.from_data"
        if not isinstance(handle, _C.DatasetWrapper):
            raise TypeError("Invalid type: {}".format(type(handle)))
        self._handle = handle

    def get_num_instances(self) -> int:
        return self._handle.get_num_instances()
    
    def get_max_dim(self) -> int:
        return self._handle.get_max_dim()

class Corpus(object):
    __create_key = object()

    @staticmethod
    def from_file(num_words: int, 
                  path: str, 
                  rank: int = 0, 
                  total_ranks: int = 1):
        handle = _C.CorpusWrapper(num_words, path, rank, total_ranks)
        return Corpus(Corpus.__create_key, handle)

    def __init__(self, create_key, handle: _C.CorpusWrapper):
        super(Corpus, self).__init__()
        assert(create_key == Corpus.__create_key), \
            "Corpus objects must be created by Corpus.from_file"
        if not isinstance(handle, _C.CorpusWrapper):
            raise TypeError("Invalid type: {}".format(type(handle)))
        self._handle = handle

    def get_num_docs(self) -> int:
        return self._handle.get_num_docs()
    
    def get_num_words(self) -> int:
        return self._handle.get_num_words()

    def get_num_tokens(self) -> int:
        return self._handle.get_num_tokens()

    def get_word_size(self) -> int:
        return self._handle.get_word_size()

    def get_doc_size(self) -> int:
        return self._handle.get_doc_size()
