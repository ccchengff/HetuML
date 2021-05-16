# -*- coding:utf-8 -*-

from hetuml import _C
from hetuml.model_base import MLBase, join_args, args_dict_to_string
from hetuml.data import Dataset, CSRMatrix
import numpy as np

from typing import Union, Iterable

class KNN(MLBase):

    def __init__(self, 
                 num_label: int, 
                 num_neighbor: int = 5, 
                 aux_args=dict()):
        assert num_label > 1, "Number of labels should be greater than 1"
        assert num_neighbor > 0, "Number of neighbors should be positive"
        args = join_args(
            aux_args=aux_args, 
            num_label=num_label, 
            num_neighbor=num_neighbor)
        handle = _C.KNN(args_dict_to_string(args))
        super(KNN, self).__init__(handle, args)
    
    def predict_score(self, 
                      labeled_data: Dataset, 
                      pred_data: Union[CSRMatrix, Dataset]
                      ) -> np.array:
        if not isinstance(labeled_data, Dataset):
            labeled_data = Dataset.from_data(labeled_data)
        if not isinstance(pred_data, (Dataset, CSRMatrix)):
            if isinstance(pred_data, tuple):
                pred_data = Dataset.from_data(pred_data)
            else:
                pred_data = CSRMatrix.from_data(pred_data)
        if isinstance(pred_data, CSRMatrix):
            scores = self._handle.Predict1(labeled_data._handle, pred_data._handle)
        else:
            scores = self._handle.Predict2(labeled_data._handle, pred_data._handle)
        num_label = self._args["num_label"]
        scores = scores.reshape(-1, num_label)
        return scores
    
    def predict_proba(self, 
                      labeled_data: Dataset, 
                      pred_data: Union[CSRMatrix, Dataset]
                      ) -> np.array:
        scores = self.predict_score(labeled_data, pred_data)
        tmp = np.exp(scores - np.max(scores, axis=1).reshape(-1, 1))
        proba = tmp / tmp.sum(axis=1).reshape(-1, 1)
        return proba
    
    def evaluate(self, 
                 labeled_data: Dataset, 
                 eval_data: Dataset, 
                 metrics: Union[str, Iterable[str]]
                 ) -> np.array:
        if not isinstance(labeled_data, Dataset):
            labeled_data = Dataset.from_data(labeled_data)
        if not isinstance(eval_data, Dataset):
            eval_data = Dataset.from_data(eval_data)
        if isinstance(metrics, str):
            metrics_list = metrics.split(",")
        else:
            metrics_list = list(metrics)
        metrics_value = self._handle.Evaluate0(
            labeled_data._handle, eval_data._handle, 
            metrics_list)
        return dict(zip(metrics_list, metrics_value))
