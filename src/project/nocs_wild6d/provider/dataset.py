import os

import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        pts_dir_list = ['data/nocs_wild6d/CAMERA25', 'data/nocs_wild6d/REAL275', 'data/nocs_wild6d/Wild6D']

        self.pts_list = []

        # 遍历路径列表下的文件, 将点云加载到列表
        for pts_dir in pts_dir_list:
            pts_file_list = os.listdir(pts_dir)
            for pts_file in pts_file_list:
                pts_path = os.path.join(pts_dir, pts_file)
                self.pts_list.append(np.load(pts_path))
        
        self.length = len(self.pts_list)     


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        pts = self.pts_list[index]
        if index < 2000:
            label = np.array([1, 0, 0])
        elif index < 4000:
            label = np.array([0, 1, 0])
        else:
            label = np.array([0, 0, 1])

        inputs = {}
        inputs['pts'] = torch.FloatTensor(pts)
        inputs['label'] = torch.FloatTensor(label)

        return inputs