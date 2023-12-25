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
import matplotlib.pyplot as plt

import src.network.nocs_wild6d.net as nocs_wild6d_net
import src.provider.nocs_wild6d.dataset as nocs_wild6d_dataset
import src.runner.nocs_wild6d.solver as nocs_wild6d_solver
from src.utils.visualization.tsne import tsne


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
    model = nocs_wild6d_net.Net(cfg).cuda()

    gorilla.solver.load_checkpoint(model=model, filename='log/main/epoch_60.pth')

    model = model.eval()


    # Loss

    loss = nocs_wild6d_net.Loss(cfg).cuda()
   
    # Dataset

    dataset = nocs_wild6d_dataset.Dataset(cfg)



    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False
        )
    
    label_list = []
    pts_feat_list = []
    for i, data in enumerate(dataloder):
        data['pts'] = data['pts'].cuda()
        data['label'] = data['label'].cuda()
        outputs = model(data)
        
        label = outputs['label_gt'].cpu().detach().numpy()[0]
        # label = outputs['label_pred'].cpu().detach().numpy()[0]

        if label[0] > 0:
            label = 1
        elif label[1] > 0:
            label = 2
        else:
            label = 3

        pts_feat = outputs['pts_feat'].cpu().detach().numpy()
        
        label_list.append(label)
        pts_feat_list.append(pts_feat)


    
    embedding = np.squeeze(np.array(pts_feat_list).astype(np.float64), axis=1)
    category = np.array(label_list)

    # enbedding visualization
    Y = tsne(embedding, 2, 50, 30.0)
    y_1 = Y[np.where(category == 1)[0], :]
    s_1 = plt.scatter(y_1[:, 0], y_1[:, 1], s=20, marker='o', c='tab:orange')
    y_2 = Y[np.where(category == 2)[0], :]
    s_2 = plt.scatter(y_2[:, 0], y_2[:, 1], s=20, marker='^', c='tab:blue')
    y_3 = Y[np.where(category == 3)[0], :]
    s_3 = plt.scatter(y_3[:, 0], y_3[:, 1], s=20, marker='s', c='tab:olive')

    plt.legend((s_1, s_2, s_3),
            ('CAMERA25', 'REAL275', 'Wild6D'),
            loc='best', ncol=1, fontsize=8, frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('visual_embedding.png', bbox_inches='tight')
