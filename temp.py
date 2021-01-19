import os

import torch

from resnet import resnet10

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
x = torch.ones([1, 1, 20, 448, 448], dtype=torch.float32).cuda()
net = resnet10().cuda()
cls, cams = net(x)
print(cls.shape, cams.shape)
