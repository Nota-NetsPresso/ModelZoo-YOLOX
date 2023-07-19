#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False, netspresso=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOFPN, YOLOXHead, YOLOXHead_1
            backbone = YOLOFPN()
            head = YOLOXHead_1(self.num_classes, self.width, in_channels=[128, 256, 512], act="lrelu") if netspresso else YOLOXHead(self.num_classes, self.width, in_channels=[128, 256, 512], act="lrelu")
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model
    
    def get_head(self):
        from yolox.models import YOLOXHead_2
        
        if "head" not in self.__dict__:
            self.head = YOLOXHead_2(self.num_classes, in_channels=[128, 256, 512])
        return self.head
