- python 3.9
- pytorch 1.13.0
- CUDA 11.6

conda create -n master python=3.9
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install gorilla-core==0.2.7.8

pip install opencv-python==4.6.0.66
pip install xformers==0.0.16    # dinov2 related (requires torch==1.13.1)
pip install triton==2.0.0   # dinov2 related

utils
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.0

pip install trimesh==4.0.8
pip install pyglet==1.5.0

MASTER
├── data 
│   ├── <data_a>
│   └── <data_b>
├── src
│   ├── config
│   ├── module
│   ├── network
│   ├── provider
│   ├── runner
│   └── utils
└── log

export https_proxy=127.0.0.1:7890

Code Logic

main.py & runner/*

main.py - template of train.py & test.py
func get_parser: 设置参数
func get_logger: 初始化日记记录(文件 + 控制台)
func init: 调用以上两个函数进行初始化

runner
├── <experiment_a> 
│   ├── train.py
│   ├── test.py
│   └── solver.py
└── <experiment_b> 
    ├── train.py
    ├── test.py
    └── solver.py

