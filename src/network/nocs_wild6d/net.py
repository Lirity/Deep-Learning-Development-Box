import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.extractor.pts.pointnet_lib.pointnet2 import Pointnet2ClsMSG

# class Net(nn.Module):
#     def __init__(self, cfg):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(3, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(256, 256, 1)
#         self.conv4 = nn.Conv1d(256, 1024, 1)
#         self.fc = nn.Linear(1024, cfg.emb_dim)

#         self.fc1 = nn.Linear(cfg.emb_dim, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 3)

#     def forward(self, inputs):
#         pts = inputs['pts']
#         c = torch.mean(pts, 1, keepdim=True)
#         pts = pts - c
#         N = pts.size()[1]

#         x = F.relu(self.conv1(pts.permute(0, 2, 1)))
#         x = F.relu(self.conv2(x))
#         global_feat = F.adaptive_max_pool1d(x, 1)
#         x = torch.cat((x, global_feat.repeat(1, 1, N)), dim=1)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)
#         pts_feat = self.fc(x)

#         out = F.relu(self.fc1(pts_feat))
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
#         out = F.softmax(out, -1)

#         outputs = {}
#         outputs['pts_feat'] = pts_feat
#         outputs['label_pred'] = out
#         outputs['label_gt'] = inputs['label']

#         return outputs

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.pts_encoder = Pointnet2ClsMSG(0)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, inputs):
        pts = inputs['pts']
        c = torch.mean(pts, 1, keepdim=True)
        pts = pts - c
        N = pts.size()[1]

        pts_feat = self.pts_encoder(pts)

        out = F.relu(self.fc1(pts_feat))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.softmax(out, -1)

        outputs = {}
        outputs['pts_feat'] = pts_feat
        outputs['label_pred'] = out
        outputs['label_gt'] = inputs['label']

        return outputs

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()


    def forward(self, inputs):
        label_gt = inputs['label_gt']
        label_pred = inputs['label_pred']
        loss = self.loss(label_gt, label_pred)

        return {'loss': loss}