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

from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


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

    """
        Fine-tuning arguments
    """
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        default="tensorboard"
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

    """ 
        Model compression - recommendation compression 
    """
    logger.info("Compression step start.")
    
    compressor = ModelCompressor(email=args.np_email, password=args.np_password)

    UPLOAD_MODEL_NAME = "yolox_model"
    TASK = Task.OBJECT_DETECTION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = exp.exp_name + '_fx.pt'
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": exp.input_size}]
    model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )

    COMPRESSED_MODEL_NAME = "test_l2norm"
    COMPRESSION_METHOD = CompressionMethod.PR_L2
    RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
    RECOMMENDATION_RATIO = 0.6
    OUTPUT_PATH = exp.exp_name + '_compressed.pt'
    compressed_model = compressor.recommendation_compression(
        model_id=model.model_id,
        model_name=COMPRESSED_MODEL_NAME,
        compression_method=COMPRESSION_METHOD,
        recommendation_method=RECOMMENDATION_METHOD,
        recommendation_ratio=RECOMMENDATION_RATIO,
        output_path=OUTPUT_PATH,
    )

    logger.info("Compression step end.") 
    
    """ 
        Retrain YOLOX model 
    """
    logger.info("Fine-tuning step start.")
    compressed_path = OUTPUT_PATH
    head_path = exp.exp_name + '_head.pt'
    
    exp = get_exp(args.exp_file, args.name + '-netspresso')
    check_exp_value(exp)
    exp.merge(args.opts)

    exp.compressed_model = compressed_path
    exp.head = head_path
    model = exp.get_model()
    model.train()

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )

    logger.info("Fine-tining step end.")
