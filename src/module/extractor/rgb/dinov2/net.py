import torch
import torch.nn as nn

from torchvision import transforms

# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') Feature Dim = 384
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') Feature Dim = 768
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') Feature Dim = 1024
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') Feature Dim = 1536

# x_norm_clstoken: [batch, feat_dim]
# x_norm_regtokens: [batch, 0, feat_dim]
# x_norm_patchtokens: [batch, patch*patch, feat_dim]
# x_prenorm: [batch, 1+patch*patch, feat_dim]

'''
利用大模型DINOv2提取图像特征

Parameters:
-rgb (torch.float32): 维度为[batch, 3, h, w]的图像, 除以255.0进行了归一化处理

Returns:
-dino_feature (torch.float32): 直接可视化出来
'''
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.extractor_preprocess = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

    def forward(self, inputs):
        rgb = inputs['rgb'] 
        
        rgb = self.extractor_preprocess(rgb)

        with torch.no_grad():
            dino_feature = self.extractor.forward_features(rgb)["x_prenorm"][:, 1:]
        dino_feature = dino_feature.reshape(dino_feature.shape[0], self.cfg.num_patches, self.cfg.num_patches, -1).contiguous() 
        
        outputs = {'dino_feature': dino_feature}
        
        return outputs
