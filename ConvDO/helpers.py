#usr/bin/python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence
import numpy as np
from inspect import isfunction
from .meta_type import *


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d