#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import spx_info_map
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import random

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
        out_L1 = self.conv1(x)
        out_L1 = self.layer1(out_L1)
        out_L4 = self.layer2(out_L1)
        out_L4 = self.layer3(out_L4)
        out_L4 = self.layer4(out_L4)
        
        return out_L4, out_L1

class SPX_embedding(nn.Module):
    def __init__(self, config, in_ch=3):
        super(SPX_embedding, self).__init__()
        
        if config.feat_extractor_backbone.lower() == 'resnet18':
            self.feat_extractor = ResNet18(in_ch) # img
        
        self.post_convolution = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), 
                                              nn.BatchNorm2d(64),nn.ReLU())
        
        self.deconv_part1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, 
                                         kernel_size=5, stride=2, padding=2, 
                                         output_padding=1, groups=1, bias=True, 
                                         dilation=1)
        self.deconv_part2 = nn.Sequential(nn.BatchNorm2d(64),nn.ReLU())
        
        self.conv_1x1 = nn.Sequential(nn.Conv2d(in_channels=67, out_channels=config.sf_channels,
                                  kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(config.sf_channels),nn.ReLU())
    
    def forward(self, img, spx):
        res_feat_L4, res_feat_L1 = self.feat_extractor(img) # L4=[b,256,w/4,h/4], L1=[b,64,w/2,h/2]
        res_feat_L4 = self.post_convolution(res_feat_L4)
        
        # (w, h) sized feats
        deconv_feat = self.deconv_part1(res_feat_L1, output_size=(img.shape[2], img.shape[3]))
        deconv_feat = self.deconv_part2(deconv_feat)
        info_map = spx_info_map(spx)
        feat = torch.cat((deconv_feat, info_map),1)
        feat = self.conv_1x1(feat)
                
        # (w/4, h/4) sized feats
        _, _, w, h = res_feat_L4.shape
        small_spx = F.interpolate(spx, (w, h), mode='nearest')
        small_info_map = spx_info_map(small_spx)
        small_feat = torch.cat((res_feat_L4, small_info_map),1)
        small_feat = self.conv_1x1(small_feat)
        
        return feat, spx, small_feat, small_spx


class My_Net(nn.Module):
    def __init__(self, config):
        super(My_Net, self).__init__()
        
        self.spx_emb = SPX_embedding(config)
        
        self.sf_conv_1x1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0),
                                         nn.BatchNorm2d(1), nn.ReLU())        
        self.f_size = config.f_size
        self.k_size = config.k_size
        
        self.sf_conv1d = nn.Sequential(
              nn.Conv1d(1, 1, self.k_size, stride=self.k_size), nn.BatchNorm1d(1), nn.ReLU(),
              nn.Conv1d(1, 1, self.k_size, stride=self.k_size), nn.BatchNorm1d(1), nn.ReLU(),
              nn.Conv1d(1, 1, self.k_size, stride=self.k_size), nn.BatchNorm1d(1), nn.ReLU())
        #self.sf_conv1d = None
    
    def get_super_feat(self, feat, spx, small_feat, small_spx):
    
        batch_size, _, _, _ = feat.shape
        sf = []
        
        for b in range(batch_size):
            big_sf, num_spx = self.frame_super_feat2(feat[b], spx[b])
            small_sf, _ = self.frame_super_feat2(small_feat[b], small_spx[b], num_spx)
            sf.append(self.sf_conv_1x1(torch.cat([big_sf[None,None], small_sf[None,None]], dim=1)).squeeze())            
        return sf
    
    def frame_super_feat(self, frame_feat, frame_spx, num_spx=None):

        if num_spx is None:
            num_spx = frame_spx.max().int()
        
        c, w, h = frame_feat.shape
        frame_spx = frame_spx.expand(c, w, h).int()
        super_feat = torch.zeros((num_spx, c*2), device=frame_feat.device)
        
        for n in range(num_spx):            
            if torch.count_nonzero(frame_spx[0] == n+1):              
                super_feat[n,0::2] = frame_feat[frame_spx==n+1].view(c,-1).mean(-1)                
                super_feat[n,1::2] = frame_feat[frame_spx==n+1].view(c,-1).std(-1, unbiased=False)
                #super_feat[n] = frame_feat[frame_spx==n+1].view(c,-1).mean(-1)                                      
            else:
                super_feat[n] = 0.0 
        
        return super_feat, num_spx

    
    def frame_super_feat2(self, frame_feat, frame_spx, num_spx=None):
        
        if num_spx is None:
            num_spx = frame_spx.max().int()
        
        c, w, h = frame_feat.shape
        num_samples = self.f_size * (self.k_size ** 4)
        frame_spx = frame_spx.expand(c, w, h).int()
        super_feat = torch.zeros((num_spx, self.f_size * c * 3), device=frame_feat.device)
        
        for n in range(num_spx):
            pop = frame_feat[frame_spx==n+1]
            pop_size = int(pop.shape[0]/c)
            
            if pop_size > 0:                
                if pop_size >= num_samples: 
                    idx = torch.LongTensor(random.sample(range(pop_size), num_samples)).to(device=frame_feat.device)#.long()
                else:
                    idx = torch.LongTensor(random.choices(range(pop_size), k=num_samples)).to(device=frame_feat.device)#.long()
                
                all_idx = torch.LongTensor([]).to(device=frame_feat.device)#.long()                        
                for i in range(c):
                    all_idx = torch.cat((all_idx, idx+(pop_size*i)))
                
                x = torch.take(pop, all_idx)
                x = self.sf_conv1d(x.view(1,1,-1))
                super_feat[n] = x                
            else:
                super_feat[n] = 0.0 
            
        return super_feat, num_spx

    
    def forward(self, img, spx):
        
        # feature embedding by superpixel
        *spx_emb, = self.spx_emb(img, spx)
        
        # make super features from superpixels and embeddings
        return self.get_super_feat(*spx_emb)
   

def compute_loss(loss_fun, spx_pools, super_feat, t_per_anchor=10):
    
    batch_size = len(spx_pools)
    loss = 0
    
    for b in range(batch_size):
        
        indices_tuple = lmu.get_random_triplet_indices(spx_pools[b], t_per_anchor=t_per_anchor)
        
        loss += loss_fun(super_feat[b], spx_pools[b], indices_tuple)
    
    return loss/batch_size


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
    if config.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
    return optimizer, lr_scheduler