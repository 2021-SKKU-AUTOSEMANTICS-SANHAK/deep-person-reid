# -*- coding: utf-8 -*-
# @Date    : 2021-01-25 22:50:29
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import os, sys
import numpy as ny
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class LUPNet(nn.Module):
    def __init__(self, backbone, embed_dim=128, cls_num=1000, per_model=None):
        super(LUPNet, self).__init__()
        self.embed_dim = embed_dim
        self.cls_num   = cls_num
        self.per_model = per_model
        self.has_embed = self.embed_dim > 0
        base = backbone()
        if self.per_model is not None and os.path.isfile(self.per_model):
            print(f'Loading pre-model from {self.pre_model}')
            state_dict = torch.load(self.pre_model, map_location=torch.device('cpu'))
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            msg = base.load_state_dict(state_dict, strict=False)
            print(f'Load pre-model with MSG: {msg}')

        self.conv1    = base.conv1
        self.bn1      = base.bn1
        self.relu     = base.relu
        self.maxpool  = base.maxpool
        self.layer1   = base.layer1
        self.layer2   = base.layer2
        self.layer3   = base.layer3
        self.layer4   = base.layer4
        self.avgpool  = base.avgpool

        in_features   = base.fc.in_features
        if self.has_embed:
            self.embed_fc = nn.Linear(in_features, self.embed_dim)
            self.embed_relu = nn.ReLU()
            init.kaiming_normal_(self.embed_fc.weight, mode='fan_out')
            init.constant_(self.embed_fc.bias, 0.)
            in_features = self.embed_dim

        self.fc = nn.Linear(in_features, self.cls_num)
        init.normal_(self.fc.weight, std=0.001)
        init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)

        if not self.training:
            return F.normalize(x)

        if self.has_embed:
            x = self.embed_fc(x)
            x = self.embed_relu(x)

        x = self.fc(x)

        return x

