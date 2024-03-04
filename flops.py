#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-4-11 21:21
# @Author  : 26731
# @File    : flops.py
# @Software: PyCharm


import torch
import thop
from thop import profile

# from models import UNet, FFDNet, DnCNN, IRCNN, UNet_Atten_4, FastDerainNet, DRDNet, EAFN
from models import FFDNet, DnCNN, IRCNN, HN, FastDerainNet, DRDNet, EAFN
from torch.autograd import Variable
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
noise_sigma = 0

noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(1)]))
noise_sigma = Variable(noise_sigma)
noise_sigma = noise_sigma.cuda()
# net = UNet_Atten_4()  # 定义好的网络模型
# net = HN()
# net = EAFN()
# net = FastDerainNet(3, 48)
net = DRDNet(3,48)
net = net.cuda()
input = torch.randn(1, 3, 256, 256)
input = input.cuda()
flops, params = profile(net, (input,))
flops, params = thop.clever_format([flops, params], "%.3f")  # 提升结果可读性
print('flops: ', flops, 'params: ', params)
