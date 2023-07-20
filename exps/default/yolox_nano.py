#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False, netspresso=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, YOLOXHead_1, TEMPbind
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            if netspresso:
                head = YOLOXHead_1(
                    self.num_classes, self.width, in_channels=in_channels,
                    act=self.act, depthwise=True
                )
            else:
                head = YOLOXHead(
                    self.num_classes, self.width, in_channels=in_channels,
                    act=self.act, depthwise=True
                )
            self.model = TEMPbind(backbone, head) if netspresso else YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    
    def get_head(self):
        from yolox.models import YOLOXHead_2
        
        if "head" not in self.__dict__:
            in_channels = [256, 512, 1024]
            self.head = YOLOXHead_2(self.num_classes, in_channels=in_channels)
        return self.head
