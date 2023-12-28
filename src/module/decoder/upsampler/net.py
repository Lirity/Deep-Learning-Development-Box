import torch
import torch.nn as nn
import torch.nn.functional as F

'''
利用大模型DINOv2提取图像特征图尺寸较小, 利用这个模块可以映射到原图尺寸

Parameters:
-rgb_feat (torch.float32): 维度为[batch, c, h, w]的特征图
-size: 上采样后的特征图尺寸[H, W]

Returns:
-rgb_feat (torch.float32): 维度为[batch, c, H, W]的特征图
'''
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.UpSampling_1 = UpSampling()
        self.UpSampling_2 = UpSampling()
        self.UpSampling_3 = UpSampling()
        self.UpSampling_4 = UpSampling()

    def forward(self, rgb_feat, size = [[28, 28], [56, 56], [112, 112], [224, 224]]): 
        rgb_feat = self.UpSampling_1(rgb_feat, size[0])
        rgb_feat = self.UpSampling_2(rgb_feat, size[1])
        rgb_feat = self.UpSampling_3(rgb_feat, size[2])
        rgb_feat = self.UpSampling_4(rgb_feat, size[3])
        return rgb_feat

class UpSampling(nn.Module):
    def __init__(self, in_channels=384, out_channels=196):
        super(UpSampling, self).__init__()
        self.TransConv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Conv = nn.Conv2d(in_channels=in_channels+out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, rgb_feat, size): 
        x = F.interpolate(rgb_feat, size=size, mode="bilinear", align_corners=True)
        rgb_feat = self.TransConv(rgb_feat)
        rgb_feat = torch.cat((rgb_feat, x), dim=1)
        rgb_feat = self.relu(self.bn(self.Conv(rgb_feat)))
        return rgb_feat


if __name__ == "__main__":
    model = Net(None)
    rgb_feat = torch.randn(48, 384, 14, 14)
    outputs = model(rgb_feat)
    print(outputs.shape)
