# -*- coding:utf-8 -*-

from hetuml import _C
from hetuml.model_base import MLBase, join_args, args_dict_to_string
from hetuml.data import COOMatrix
import numpy as np

from typing import Optional

class NMF(MLBase):

    def __init__(self, 
                 embedding_dim: int = 8, 
                 num_epoch: int = 10, 
                 learning_rate: float = 0.1, 
                 l1_reg: float = 0.0, 
                 l2_reg: float = 0.0, 
                 aux_args=dict()):
        assert embedding_dim > 0, "Embedding dimemsion should be positive"
        assert num_epoch > 0, "Number of epochs should be positive"
        assert learning_rate > 0, "Learning rate should be positive"
        assert l1_reg >= 0, "L1 regularization term should be non-negative"
        assert l2_reg >= 0, "L2 regularization term should be non-negative"
        args = join_args(
            aux_args=aux_args, 
            embedding_dim=embedding_dim, 
            num_epoch=num_epoch,
            learning_rate=learning_rate,
            l1_reg=l1_reg,
            l2_reg=l2_reg)
        handle = _C.MF(args_dict_to_string(args))
        super(NMF, self).__init__(handle, args)
    
    def fit(self, 
            train_data: COOMatrix, 
            valid_data: Optional[COOMatrix] = None
            ) -> None:
        if not isinstance(train_data, COOMatrix):
            train_data = COOMatrix.from_data(train_data)
        if valid_data is not None:
            if not isinstance(valid_data, COOMatrix):
                valid_data = COOMatrix.from_data(valid_data)
        if valid_data is None:
            self._handle.Fit1(train_data._handle)
        else:
            self._handle.Fit2(train_data._handle, valid_data._handle)

    def predict(self, pred_data: COOMatrix) -> np.array:
        if not isinstance(pred_data, COOMatrix):
            pred_data = COOMatrix.from_data(pred_data)
        return self._handle.Predict1(pred_data._handle)
