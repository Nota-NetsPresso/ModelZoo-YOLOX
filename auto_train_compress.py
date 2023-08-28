#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")

    """
        Common arguments
    """
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    """
        Compression arguments
    """
    parser.add_argument(
        "-w",
        "--weight_path",
        type=str
    )
    parser.add_argument(
        "-m",
        "--np_email",
        help="NetsPresso login e-mail",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--np_password",
        help="NetsPresso login password",
        type=str,
    )

    return parser


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()

    """ 
        Convert YOLOX model to fx 
    """
    logger.info("yolox to fx graph start.")

    exp = get_exp(args.exp_file, args.name)
    check_exp_value(exp)
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model(netspresso=True)

    # load the model state dict
    ckpt = torch.load(args.weight_path, map_location="cpu")['model']

    model.train()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)

    logger.info("loading checkpoint done.")
    
    _graph = torch.fx.Tracer().trace(model)
    traced_model = torch.fx.GraphModule(model, _graph)
    torch.save(traced_model, exp.exp_name + '_fx.pt')
    logger.info(f"generated model to compress model {os.path.join(exp.output_dir, exp.exp_name, exp.exp_name + '_fx.pt')}")
    
    head = exp.get_head()
    torch.save(head, exp.exp_name + '_head.pt')
    logger.info(f"generated model to model's head {os.path.join(exp.output_dir, exp.exp_name, exp.exp_name + '_head.pt')}")

    logger.info("yolox to fx graph end.")
