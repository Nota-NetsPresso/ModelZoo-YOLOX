#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

class TEMPbind(nn.Module):
    """
    This is a temporary binding class to export YOLOX for NetsPresso.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        x = self.backbone(x)
        outputs = self.head(x)
        
        return outputs
