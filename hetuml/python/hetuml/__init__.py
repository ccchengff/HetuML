# -*- coding:utf-8 -*-

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import importlib as _importlib

def _import_module(mod_name):
    return _importlib.import_module(mod_name)

# globally accessible backend library
_C = _import_module("hetuml_core")
