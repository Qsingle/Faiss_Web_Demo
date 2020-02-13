#-*- coding:utf8 -*-
#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

from .ops import get_layers,BasicBlock,BottleBlock,Conv2d

class ResNet(nn.Module):
    def __init__(self, in_ch, num_classes=3, n_layers=18):
        super(ResNet, self).__init__()
        ch = 64
        self.conv1 = Conv2d(in_ch, ch, 3, 1)
        self.conv2 = Conv2d(ch, ch, 3, 2, padding=1)
        self.conv3 = Conv2d(ch, ch, 3, 1)
        self.pool = nn.AvgPool2d(7, 1)

        if n_layers < 50:
            Block = BasicBlock
        else:
            Block = BottleBlock

        self.layers = get_layers(n_layers)
        self.block1 = Block(ch, ch,strides=2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block1_x = Block(in_cha, ch)
        if n_layers < 50:
            chb = ch
        else:
            chb = ch  * 4
        ch*=2
        self.block2 = Block(chb, ch,2,downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block2_x = Block(in_cha, ch)
        if n_layers < 50:
            chb = ch
        else:
            chb = ch * 4
        ch*=2
        self.block3 = Block(chb,ch, 2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block3_x = Block(in_cha, ch)
        if n_layers < 50:
            chb = ch
        else:
            chb = ch * 4
        ch*=2
        self.block4 = Block(chb,ch,2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block4_x = Block(in_cha, ch)
        inc = ch
        if n_layers >= 50:
            inc = ch*4
        self.fc = nn.Linear(in_features=inc,out_features=num_classes)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.block1(net)
        for i in range(1,self.layers[0]):
            net = self.block1_x(net)
        net = self.block2(net)
        for i in range(1, self.layers[1]):
            net = self.block2_x(net)
        net = self.block3(net)
        for i in range(1,self.layers[2]):
            net = self.block3_x(net)
        net = self.block4(net)
        for i in range(1,self.layers[3]):
            net = self.block4_x(net)
        net = self.pool(net)
        net = net.view(net.size(0),-1)
        net = self.fc(net)
        return net

if __name__ == "__main__":
    model = ResNet(3,3,50)
    import numpy as np 
    np.random.seed(65535)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    x = np.random.normal(0,1,[1,3,224,224])
    x = torch.Tensor(x)
    x = x.to(device)
    out = model(x)
    out = out.detach().cpu().numpy()
    print(out)