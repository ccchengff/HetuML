# -*- coding:utf-8 -*-

from hetuml import _C
from hetuml.model_base import SupervisedMLBase, join_args, args_dict_to_string

class SupportVectorMachine(SupervisedMLBase):

    def __init__(self, 
                 num_epoch: int = 10, 
                 batch_size: int = 1000, 
                 learning_rate: float = 0.1, 
                 l1_reg: float = 0.0, 
                 l2_reg: float = 0.0, 
                 loss: str = "hinge", 
                 metrics: str = "hinge-loss", 
                 aux_args=dict()):
        assert num_epoch > 0, "Number of epochs should be positive"
        assert batch_size > 0, "Batch size should be positive"
        assert learning_rate > 0, "Learning rate should be positive"
        assert l1_reg >= 0, "L1 regularization term should be non-negative"
        assert l2_reg >= 0, "L2 regularization term should be non-negative"
        assert loss in ("hinge",), "Loss {} is not allowed for {}".format(loss, type(self))
        args = join_args(
            aux_args=aux_args, 
            use_neg_y=True, 
            num_epoch=num_epoch, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            l1_reg=l1_reg, 
            l2_reg=l2_reg, 
            loss=loss, 
            metrics=metrics)
        handle = _C.SVM(args_dict_to_string(args))
        super(SupportVectorMachine, self).__init__(handle, args)
