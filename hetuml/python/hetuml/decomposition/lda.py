# -*- coding:utf-8 -*-

from hetuml import _C
from hetuml.model_base import MLBase, join_args, args_dict_to_string
from hetuml.data import Corpus
import numpy as np

class LDA(MLBase):
    ps_data_type = "int"

    def __init__(self, 
                 num_words: int, 
                 num_topics: int = 100, 
                 num_iters: int = 100, 
                 alpha: float = 0.0, 
                 beta: float = 0.001, 
                 long_doc_thres: int = 600, 
                 parallel=False, 
                 aux_args=dict()):
        assert num_words > 0, "Number of words should be positive"
        assert num_topics > 0, "Number of topics should be positive"
        assert num_iters > 0, "Number of iterations should be positive"
        assert alpha >= 0, "Alpha should be non-negative"
        assert beta >= 0, "Alpha should be non-negative"
        assert long_doc_thres >= 0, "Threshold for long docs should be non-negative"
        args = join_args(
            aux_args=aux_args, 
            num_topics=num_topics, 
            num_words=num_words,
            num_iters=num_iters,
            alpha=alpha,
            beta=beta,
            long_doc_thres=long_doc_thres, 
            parallel=parallel)
        if parallel:
            handle = _C.ParallelLDA(args_dict_to_string(args))
        else:
            handle = _C.LDA(args_dict_to_string(args))
        super(LDA, self).__init__(handle, args)
    
    def fit(self, train_data: Corpus) -> None:
        if not isinstance(train_data, Corpus):
            raise TypeError("Input for {} should be Corpus"
                            .format(self.name()))
        self._handle.Fit1(train_data._handle)

    def predict(self, pred_data: Corpus) -> np.array:
        if not isinstance(pred_data, Corpus):
            raise TypeError("Input for {} should be Corpus"
                            .format(self.name()))
        return self._handle.Predict1(pred_data._handle)
