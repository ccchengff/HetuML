# -*- coding:utf-8 -*-

import numpy as np
from hetuml.data import Dataset, CSRMatrix
from hetuml import _C

from typing import Union, Tuple, Iterable, Optional
from copy import deepcopy

class MLBase(object):
    
    def __init__(self, handle, args: dict):
        super(MLBase, self).__init__()
        self._handle = handle
        self._args = args
    
    def save_model(self, path: str) -> None:
        self._handle.SaveModel(path)
    
    def load_model(self, path: str) -> None:
        self._handle.LoadModel(path)
        loaded_args = self._handle.GetParams()
        for name, value in loaded_args.items():
            name = name.lower()
            if name in self._args:
                old_value = self._args[name]
                if isinstance(old_value, bool):
                    self._args[name] = True if value == "true" else False
                elif isinstance(old_value, int):
                    self._args[name] = int(value)
                elif isinstance(old_value, float):
                    self._args[name] = float(value)
                else:
                    self._args[name] = value
            else:
                self._args[name] = value
    
    def name(self) -> str:
        return self._handle.name()
    
    def __getitem__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        elif name in self._args:
            return self._args[name]
        else:
            raise AttributeError("Type {} does not have attribute {}"
                                 .format(type(self), name))


class SupervisedMLBase(MLBase):

    def __init__(self, handle, args: dict):
        super(SupervisedMLBase, self).__init__(handle, args)
    
    def fit(self, 
            train_data: Dataset, 
            valid_data: Optional[Dataset] = None
            ) -> None:
        if not isinstance(train_data, Dataset):
            train_data = Dataset.from_data(train_data)
        if valid_data is not None:
            if not isinstance(valid_data, Dataset):
                valid_data = Dataset.from_data(valid_data)
        if valid_data is None:
            self._handle.Fit1(train_data._handle)
        else:
            self._handle.Fit2(train_data._handle, valid_data._handle)
    
    def predict_score(self, 
                      pred_data: Union[Dataset, CSRMatrix]
                      ) -> np.array:
        if not isinstance(pred_data, (Dataset, CSRMatrix)):
            if isinstance(pred_data, tuple):
                pred_data = Dataset.from_data(pred_data)
            else:
                pred_data = CSRMatrix.from_data(pred_data)
        if isinstance(pred_data, CSRMatrix):
            scores = self._handle.Predict1(pred_data._handle)
        else:
            scores = self._handle.Predict2(pred_data._handle)
        if self.is_classification():
            num_label = self.get_num_label()
            if num_label > 2:
                scores = scores.reshape(-1, num_label)
        return scores
    
    def predict_proba(self, 
                      pred_data: Union[Dataset, CSRMatrix]
                      ) -> np.array:
        scores = self.predict_score(pred_data)
        if self.is_regression():
            return scores
        else:
            if scores.ndim == 1:
                # sigmoid for binary-classification
                proba = 1 / (1 + np.exp(-scores))
            else:
                # softmax for multi-classification
                num_label = scores.shape[1]
                tmp = np.exp(scores - np.max(scores, axis=1).reshape(-1, 1))
                proba = tmp / tmp.sum(axis=1).reshape(-1, 1)
            return proba
    
    def evaluate(self, 
                 eval_data: Dataset, 
                 metrics: Union[str, Iterable[str]]
                 ) -> np.array:
        if not isinstance(eval_data, Dataset):
            eval_data = Dataset.from_data(eval_data)
        if isinstance(metrics, str):
            metrics_list = metrics.split(",")
        else:
            metrics_list = list(metrics)
        metrics_value = self._handle.Evaluate0(
            eval_data._handle, metrics_list)
        return dict(zip(metrics_list, metrics_value))
    
    def use_neg_y(self) -> bool:
        return self._args.get("use_neg_y", False)
    
    def is_regression(self) -> bool:
        return self._args.get("is_regression", False)
    
    def is_classification(self) -> bool:
        return not self.is_regression()
    
    def get_num_label(self) -> int:
        return self._args.get("num_label", 2)


def join_args(aux=dict(), **kwargs):
    from copy import deepcopy
    res = deepcopy(aux)
    for name, value in kwargs.items():
        res[name] = value
    return res

def args_dict_to_string(args_dict):
    res = {}
    for name, value in args_dict.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        res[str(name).upper()] = str(value)
    return res
