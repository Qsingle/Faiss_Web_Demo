#-*- coding:utf8 -*-
#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
from torch.nn import functional as F


def get_layers(n):
    layers = []
    if n == 18:
        layers = [2,2,2,2]
    elif n == 32:
        layers = [3,4,6,3]
    elif n == 50:
        layers = [3,4,6,3]
    elif n == 101:
        layers = [3,4,23,3]
    elif n == 152:
        layers = [3,8,36,3]
    return layers

class Conv2d(nn.Module):
    '''
    The basic conv op, include the BatchNormalization and ReLU
    '''
    def __init__(self,in_ch,out_ch,ksize,strides=1,padding=0,activation=True):
        super(Conv2d,self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=ksize,stride=strides,
        padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        net = self.conv(x)
        net = self.bn(net)
        if self.activation:
            net = self.relu(net)
        return net

class BasicBlock(nn.Module):
    '''
    ResNet's BasicBlock
    '''
    def __init__(self,in_ch,ch,strides=1,downsample=False):
        super(BasicBlock,self).__init__()
        self.conv_pre = Conv2d(in_ch,ch,3,1,padding=1)
        self.conv = Conv2d(ch,ch,3,strides,padding=1)
        self.downsample = downsample
        self.down = Conv2d(in_ch,ch,1,strides,activation=False)
    
    def forward(self,x):
        net = self.conv_pre(x)
        net = self.conv(net)
        if self.downsample:
            x = self.down(x)
        net = net + x
        net = F.relu(net)
        return net
    
class BottleBlock(nn.Module):
    def __init__(self,in_ch,ch,strides=1,downsample=False):
        super(BottleBlock,self).__init__()
        self.conv1_pre = Conv2d(in_ch,ch,1)
        self.conv3 = Conv2d(ch,ch, 3, strides, padding=1)
        self.conv1_af = Conv2d(ch,ch*4, 1, 1)
        self.downsample = downsample
        self.down = Conv2d(in_ch,ch*4,1,strides,activation=False)
    
    def forward(self,x):
        net = self.conv1_pre(x)
        net = self.conv3(net)
        net = self.conv1_af(net)
        if self.downsample:
            x = self.down(x)
        net = x + net
        net = F.relu(net)
        return net