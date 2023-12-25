import os
import random

import gorilla
import argparse
import logging

import cv2
import torch
import numpy as np

import src.module.extractor.rgb.dinov2.net as dinov2

from src.utils.document.pkl_api import read_pkl
from src.utils.visualization.feat_api import vis_feature_map
from src.utils.visualization.pts_api import vis_numpy_pts

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
    cfg.log_dir = 'log/main'
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    logger = get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir + "/training_logger.log")

    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)   # 使用指定的GPU

    return logger, cfg

def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

if __name__ == "__main__":
    vis_numpy_pts(np.load('data/nocs_wild6d/Wild6D/0000.npy'))
    aaa

    # print(read_pkl('data/nocs/Real/test/scene_3/0524_label.pkl'))
    # aaa

    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    # 设置随机种子
    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    logger.info("=> creating model ...")
    model = dinov2.Net(cfg)

    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()

    # 统计模型参数
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))

    # [137, 370, 203, 434],
    #    [ 93,  17, 324, 310],
    #    [121, 445, 210, 503],
    #    [127, 264, 228, 372],
    #    [314, 106, 433, 242]
    rmin, rmax, cmin, cmax = get_bbox([68, 431, 131, 499])

    rgb = cv2.imread('data/nocs/Real/test/scene_3/0524_color.png')[rmin:rmax, cmin:cmax, :].astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, dsize=(cfg.num_patches*14, cfg.num_patches*14), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Image Window', (rgb * 255.0).astype(np.uint8))
    cv2.waitKey(0)

    inputs = {}
    inputs['rgb'] = torch.FloatTensor(rgb).permute(2, 0, 1).cuda()
    inputs['rgb'] = inputs['rgb'].unsqueeze(0)

    outputs = model(inputs)

    print(outputs['dino_feature'].dtype)

    vis_feature_map(outputs['dino_feature'][0].cpu().detach().numpy())
