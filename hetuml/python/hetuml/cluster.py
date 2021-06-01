# -*- coding:utf-8 -*-

from hetuml import _C
import os

from typing import Union, Tuple, Iterable, Optional

class Cluster(object):
    _ROLES = ("scheduler", "server", "worker")

    def __init__(self, 
                 scheduler: str, 
                 num_servers: int, 
                 num_workers: int, 
                 role: str):
        assert role in Cluster._ROLES, \
            "Unrecognizable role: {}, should be one of {}".format(
                role, Cluster._ROLES)
        splits = scheduler.split(":")
        assert len(splits) == 2, "Invalid address: " + scheduler
        self.scheduler_ip, self.scheduler_port = splits
        self.role = role
        self.num_servers = num_servers
        self.num_workers = num_workers
        
        self._setup_env()
        _C.InitPS()
        if self.role != "scheduler":
            self.rank = _C.MyRank()
        else:
            self.rank = 0
    
    def start_server(self, data_type):
        if self.role == "server":
            data_type = data_type.lower()
            if data_type == "int":
                _C.StartIntServer()
            elif data_type == "float":
                _C.StartFloatServer()
            elif data_type == "double":
                _C.StartDoubleServer()
            else:
                raise Exception(
                    "Data type {} is not supported".format(
                        data_type))
    
    def join(self):
        _C.FinalizePS()
    
    def _setup_env(self):
        os.environ["DMLC_PS_ROOT_URI"] = self.scheduler_ip
        os.environ["DMLC_PS_ROOT_PORT"] = self.scheduler_port
        os.environ["DMLC_NUM_SERVER"] = str(self.num_servers)
        os.environ["DMLC_NUM_WORKER"] = str(self.num_workers)
        os.environ["DMLC_ROLE"] = self.role
