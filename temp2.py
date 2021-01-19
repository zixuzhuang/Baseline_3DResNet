import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import Size
from torch.autograd import Variable

import config

shape = (10,10,10)
_fn = nn.Upsample(size=shape, mode="trilinear", align_corners=True)
imgs = _fn(torch.ones([2,3,2,2,2], dtype=torch.float32))
print(imgs)
