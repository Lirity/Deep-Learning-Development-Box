import os
import sys
import random

BASE_DIR = '/data4/lj/master'
sys.path.append(BASE_DIR)

import gorilla
import argparse
import logging

import cv2
import torch
import numpy as np

import src.network.nocs_wild6d.net as nocs_wild6d_net
import src.provider.nocs_wild6d.dataset as nocs_wild6d_dataset
import src.runner.nocs_wild6d.solver as nocs_wild6d_solver


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus", type=str, default="0", help="gpu num")
    parser.add_argument("--config", type=str, default="src/config/base.yaml", help="path to config file")

    args_cfg = parser.parse_args()

    return args_cfg

def get_logger(level_print, level_save, path_file, name_logger="logger"):
    logger = logging.getLogger(name_logger) # 创建日志记录器
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')  # 格式为: 时间-消息

    # 设置文件处理器
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)   # logging.WARN
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)

    # 设置控制台处理器
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print) # logging.INFO 
    logger.addHandler(handler_view)

    return logger

def init():
    args = get_parser() # 读取局部参数

    cfg = gorilla.Config.fromfile(args.config)  # 读取配置文件(全局参数)

    cfg.gpus = args.gpus

    # 指定日志记录地址
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    logger = get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir + "/training_logger.log")

    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)   # 使用指定的GPU

    return logger, cfg

if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    # 设置随机种子
    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # Model

    logger.info("=> creating model ...")
    model = nocs_wild6d_net.Net(cfg)

    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()

    # 统计模型参数
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))

    # Loss

    loss = nocs_wild6d_net.Loss(cfg).cuda()
   
    # Dataset

    dataset = nocs_wild6d_dataset.Dataset(cfg)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train_dataloader.bs,
        num_workers=int(cfg.train_dataloader.num_workers),
        shuffle=cfg.train_dataloader.shuffle,
        sampler=None,
        drop_last=cfg.train_dataloader.drop_last,
        pin_memory=cfg.train_dataloader.pin_memory
    )

    dataloaders = {
        "train": dataloader,
    }

    # solver
    Trainer = nocs_wild6d_solver.Solver(model=model, loss=loss,
                     dataloaders=dataloaders,
                     logger=logger,
                     cfg=cfg)
    Trainer.solve()

    logger.info('\nFinish!\n')