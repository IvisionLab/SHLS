#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import spx_info_map
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_ch=1):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 256, 2, stride=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class SPX_embedding(nn.Module):
    def __init__(self, backbone='resnet18', in_ch=3):
        super(SPX_embedding, self).__init__()
        
        if backbone.lower() == 'resnet18':
            self.feat_extractor = ResNet18(in_ch) # img
        
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)
        
        self.deconv = nn.ConvTranspose2d(in_channels=64, out_channels=64, 
                                         kernel_size=5, stride=4, padding=1, 
                                         output_padding=1, groups=1, bias=True, 
                                         dilation=1)
        self.conv_1x1 = nn.Conv2d(in_channels=71, out_channels=30,
                                  kernel_size=1, stride=1, padding=0)
    
    def forward(self, img, spx):
        
        res_feat = self.feat_extractor(img)
        res_feat = self.post_convolution(res_feat)
        
        # normal size feats
        deconv_feat = self.deconv(res_feat)        
        info_map = spx_info_map(spx)
        feat = torch.cat((deconv_feat, img, spx, info_map),1)
        feat = self.conv_1x1(feat)
        feat = torch.tanh(feat)
                
        # small size feats
        _, _, w, h = res_feat.shape
        small_img = F.interpolate(img, (w,h), mode='bilinear', align_corners=False)
        small_spx = F.interpolate(spx, (w, h), mode='nearest')
        small_info_map = spx_info_map(small_spx)
        small_feat = torch.cat((res_feat, small_img, small_spx, small_info_map),1)
        small_feat = self.conv_1x1(small_feat)
        small_feat = torch.tanh(small_feat)
        
        return feat, spx, small_feat, small_spx
    


class My_Net(nn.Module):
    def __init__(self, config):
        super(My_Net, self).__init__()
        
        self.spx_emb = SPX_embedding(config.feat_extractor_backbone)
        self.sf_conv_1x1 = nn.Conv2d(in_channels=2, out_channels=1,
                                  kernel_size=1, stride=1, padding=0)
    
    def get_super_feat(self, feat, spx, small_feat, small_spx):
    
        batch_size, _, _, _ = feat.shape
        sf = []
        
        for b in range(batch_size):
            big_sf, num_spx = self.frame_super_feat(feat[b], spx[b])
            small_sf, _ = self.frame_super_feat(small_feat[b], small_spx[b], num_spx)
            
            sf.append(self.sf_conv_1x1(torch.cat([big_sf[None,None], small_sf[None,None]], dim=1)).squeeze())
        
        return sf
    
    def frame_super_feat(self, frame_feat, frame_spx, num_spx=None):

        if num_spx is None:
            num_spx = frame_spx.max().int()
        
        c, w, h = frame_feat.shape
        frame_spx = frame_spx.expand(c, w, h).int()
        super_feat = torch.zeros((num_spx, c), device=frame_feat.device)
        
        for n in range(num_spx):
            spx = frame_spx == n+1
            feats = frame_feat.clone()        
            feats[spx==False] = 0
            spx_numel = spx[0].sum()
            if spx_numel != 0:
                super_feat[n] = feats.view(c, w*h).sum(-1)/spx_numel
            else:
                super_feat[n] = 0
        
        return super_feat, num_spx
    
    def spx_embedding(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def super_feat(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def forward(self, *args, **kwargs):        
        if len(args) == 2:
            return self.spx_emb(*args, **kwargs)
        elif len(args) == 4:
            return self.get_super_feat(*args, **kwargs)
        else:
            return None



def compute_loss(loss_fun, spx_pools, super_feat, t_per_anchor=10):
    
    pools_sizes = [x.shape[0] for x in spx_pools]    
    split_size = len(super_feat)
    loss = 0    
    
    for i in range(split_size):
        sf_splits = torch.split(super_feat[i], pools_sizes[i::split_size])
        
        for n, sf in enumerate(sf_splits):     
            indices_tuple = lmu.get_random_triplet_indices(spx_pools[i+n*split_size], t_per_anchor=t_per_anchor)        
            loss += loss_fun(sf, spx_pools[i+n*split_size], indices_tuple)
            
    return loss/len(spx_pools)


def get_model_loss(config):
    model = None
    loss_fun = None
    
    model = My_Net(config)

    if config.loss == 'NTXentLoss':        
        loss_fun = NTXentLoss(temperature=0.07)
    
    return model, loss_fun


def get_opti_scheduler(config, model,train_loader=None):
    optimizer = None
    lr_scheduler = None
    if config.classifier_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.classifier_learning_rate)
    if config.classifier_optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.classifier_learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True)
        
    return optimizer, lr_scheduler