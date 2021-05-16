# -*- coding:utf-8 -*-

from hetuml import _C
from hetuml.model_base import SupervisedMLBase, join_args, args_dict_to_string

class NaiveBayes(SupervisedMLBase):

    def __init__(self, 
                 num_label: int, 
                 metrics: str = "cross-entropy", 
                 aux_args=dict()):
        assert num_label > 1, "Number of labels should be greater than 1"
        args = join_args(
            aux_args=aux_args, 
            num_label=num_label, 
            metrics=metrics)
        handle = _C.NaiveBayes(args_dict_to_string(args))
        super(NaiveBayes, self).__init__(handle, args)
