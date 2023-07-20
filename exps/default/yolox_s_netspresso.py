#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.compressed_model = "/path/your/compressed_model.pt"
        self.head = "/path/your/model_head.pt"
        
    def get_model(self, sublinear=False, netspresso=False):
        if "model" not in self.__dict__:
            from yolox.models import YOLOX_netspresso
            backbone = torch.load(self.compressed_model)
            head = torch.load(self.head)
            self.model = YOLOX_netspresso(backbone=backbone, head=head)
            
        if netspresso:
            return backbone

        return self.model
