#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.compressed_model = "/workspace/YOLOX_nota/obj_com.pt"
        self.head = "/workspace/YOLOX_nota/model_head.pt"
        
    def get_model(self, sublinear=False, netspresso=False):
        if "model" not in self.__dict__:
            from yolox.models import YOLOX_netspresso
            backbone = torch.load(self.compressed_model)
            head = torch.load(self.head)
            self.model = YOLOX_netspresso(backbone, head)
            
        if netspresso:
            return backbone

        return self.model
