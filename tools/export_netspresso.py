#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
import torch.fx as fx
import torch.nn as nn

from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX for NetsPresso")
    parser.add_argument(
        "--output-path", type=str, default=".", help="output name of models"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model(netspresso=True)
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.train()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)

    logger.info("loading checkpoint done.")
    
    from yolox.models import TEMPbind
    model_to_compress = TEMPbind(backbone=model.backbone, head=model.head)
    
    _graph = fx.Tracer().trace(model_to_compress)
    traced_model = fx.GraphModule(model_to_compress, _graph)
    torch.save(traced_model, os.path.join(args.output_path, 'model_to_compress.pt'))
    logger.info(f"generated model to compress model {os.path.join(args.output_path, 'model_to_compress.pt')}")
    
    head = exp.get_head()
    torch.save(head, os.path.join(args.output_path, 'model_head.pt'))
    logger.info(f"generated model to model's head {os.path.join(args.output_path, 'model_head.pt')}")


if __name__ == "__main__":
    main()
