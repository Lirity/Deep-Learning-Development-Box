# Deep Learning Development Box
---

## Requirements
---
- python 3.9
- pytorch 1.13.0
- CUDA 11.6

``` bash
conda create -n master python=3.9
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install gorilla-core==0.2.7.8

pip install opencv-python==4.6.0.66
pip install xformers==0.0.16    # dinov2 related (requires torch==1.13.1)
pip install triton==2.0.0       # dinov2 related

utils
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.0

pip install trimesh==4.0.8
pip install pyglet==1.5.0
```

## Directory Structure
---
``` bash
MASTER
├── doc # 笔记
├── data # 数据集
│   ├── <data_a>
│   └── <data_b>
├── src # 代码
│   ├── config  # 配置文件
│   │   └── <project_a> # 每个项目每次实验对应一个yaml
│   │       ├── <experiment_a>.yaml
│   │       └── <experiment_b>.yaml
│   ├── module  # 网络常用的模块, 如Encoder, 与具体实验无关
│   │   ├── <module_a>
│   │   └── <module_b>
│   ├── network
│   │   └── <project_a>
│   │       └── net.py
│   ├── provider
│   │   └── <project_a> 
│   │       └── dataset.py
│   ├── runner
│   │   └── <project_a>
│   │       ├── train.py
│   │       ├── test.py
│   │       └── solver.py
│   ├── utils
│   │   ├── document        # 文件io
│   │   ├── geometry        # 几何变换
│   │   └── visualization   # 可视化
└── log # 项目实验记录
    └── <project_a>
        ├── <experiment_a>
        └── <experiment_b>

# <project_*>为实验名称
```

## Code Logic
---
``` bash
# 库导入顺序
## 系统相关库
import os
import sys

## 框架相关库
import gorilla
import argparse
import logging

## 基础算法库
import cv2
import torch
import numpy as np

## 网络相关插件 
import <prefix>.<project>.<item> as <project>_<item>

## 工具相关插件(具体工具函数导入)
from ... import ... 

runner/train.py
func get_parser: 设置参数
func get_logger: 初始化日志记录(文件 + 控制台)
func init: 调用以上两个函数进行初始化
```

## Common Commands
---
``` bash
export https_proxy=127.0.0.1:7890   # VPN
```