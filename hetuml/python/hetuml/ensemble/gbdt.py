# -*- coding:utf-8 -*-

from hetuml import _C
from hetuml.model_base import SupervisedMLBase, join_args, args_dict_to_string

class GBDT(SupervisedMLBase):
    ps_data_type = "double"

    def __init__(self, 
                 max_depth: int = 4, 
                 max_node_num: int = -1, 
                 leafwise: bool = False, 
                 num_split: int = 10, 
                 ins_sp_ratio: float = 1.0, 
                 feat_sp_ratio: float = 1.0, 
                 is_regression: bool = False, 
                 num_label: int = 2, 
                 num_round: int = 10, 
                 learning_rate: float = 0.1, 
                 min_split_gain: float = 0.0, 
                 min_node_ins: int = 1024, 
                 min_child_weight: float = 0, 
                 reg_alpha: float = 0, 
                 reg_lambda: float = 0, 
                 max_leaf_weight: float = 0, 
                 multi_tree: bool = False, 
                 loss: str = "auto", 
                 metrics: str = "auto", 
                 parallel=False, 
                 aux_args=dict()):
        assert max_depth > 0, "Maximum tree depth should be positive"
        if max_node_num == -1:
            max_node_num = (2 ** (max_depth + 1)) - 1
        assert max_node_num > 0, "Maximum number of tree nodes should be positive"
        assert 0 < num_split <= 255, "Invalid number of splits: " + str(num_split)
        assert 0 < ins_sp_ratio <= 1, "Invalid sampling ratio for instances: " + str(ins_sp_ratio)
        assert 0 < feat_sp_ratio <= 1, "Invalid sampling ratio for features: " + str(feat_sp_ratio)
        assert num_label > 1, "Number of labels should be greater than 1"
        assert num_round > 0, "Number of rounds should be positive"
        assert learning_rate > 0, "Learning rate should be positive"
        assert min_split_gain >= 0, "Minimum split gain should be non-negative"
        assert min_node_ins >= 0, "Minimum instance per node should be non-negative"
        assert min_child_weight >= 0, "Minimum child weight should be non-negative"
        assert reg_alpha >= 0, "L1 regularization term should be non-negative"
        assert reg_lambda >= 0, "L2 regularization term should be non-negative"
        assert max_leaf_weight >= 0, "Maximum leaf weight term should be non-negative"
        if loss == "auto":
            if is_regression is True:
                loss = "square"
            else:
                loss = "logistic"
        if metrics == "auto":
            if is_regression is True:
                metrics = "mse"
            elif num_label == 2:
                metrics = "log-loss"
            else:
                metrics = "cross-entropy"
        args = join_args(
            aux_args=aux_args, 
            max_depth=max_depth, 
            max_node_num=max_node_num, 
            num_split=num_split, 
            ins_sp_ratio=ins_sp_ratio, 
            feat_sp_ratio=feat_sp_ratio, 
            is_regression=is_regression, 
            num_label=num_label, 
            num_round=num_round, 
            learning_rate=learning_rate, 
            min_split_gain=min_split_gain, 
            min_node_ins=min_node_ins, 
            min_child_weight=min_child_weight, 
            reg_alpha=reg_alpha, 
            reg_lambda=reg_lambda, 
            max_leaf_weight=max_leaf_weight, 
            multi_tree=multi_tree, 
            loss=loss, 
            metrics=metrics, 
            parallel=parallel)
        if parallel:
            handle = _C.ParallelGBDT(args_dict_to_string(args))
        else:
            handle = _C.GBDT(args_dict_to_string(args))
        super(GBDT, self).__init__(handle, args)
