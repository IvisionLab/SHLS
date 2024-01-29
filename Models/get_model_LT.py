#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_LT import spx_info_map, FaissKNeighbors, FaissKMeans, filt_outliers #, location_map #, to_onehot, 
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning import losses, miners, distances, reducers
from torchvision.ops import RoIAlign
#import os
#import matplotlib.pyplot as plt
#import torchvision.transforms as T

############################################## RESNET BACKBONE
class ResNet18(nn.Module):
    def __init__(self, in_ch=1):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),)

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
        out_L2 = self.conv1(x)
        out_L2 = self.layer1(out_L2)
        out_L4 = self.layer2(out_L2)
        out_L4 = self.layer3(out_L4)
        out_L4 = self.layer4(out_L4)
        
        return out_L4, out_L2

############################################## AUXILIAR RESNET BLOCKS
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

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride),
                                            nn.BatchNorm2d(outdim),nn.ReLU())
        
        self.conv1 = nn.Sequential(nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride),
                                            nn.BatchNorm2d(outdim),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, stride=stride),
                                            nn.BatchNorm2d(outdim),nn.ReLU())
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(r) 
        if self.downsample is not None:
            x = self.downsample(x)         
        return x + r 

############################################## FAKE SEGMENTER BLOCKS
class Mem_select_fake(nn.Module):
    def __init__(self, ):
        super(Mem_select_fake, self).__init__()
        
    def forward(self, am, L1, L4):
        _, c, h, w = L4.shape
        
        return L4.view(1, 2*c, h, w)

class Segmenter_fake(nn.Module):
    def __init__(self, config):
        super(Segmenter_fake, self).__init__()
        
        self.pred_L1 = nn.Conv2d(4, 2, kernel_size=3, padding=1, stride=1)        
    def forward(self, L1, L4, AM): # 139                
        
        return self.pred_L1(AM)
############################################## REAL SEGMENTER BLOCKS
class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Mem_select(nn.Module):
    def __init__(self, ):
        super(Mem_select, self).__init__()
        
        self.convAM = nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=2)
        self.convL2 = nn.Conv2d(67, 32, kernel_size=3, padding=1, stride=2)
        self.gate = nn.Conv2d(32, 259, kernel_size=3, padding=1, stride=2)
        
    def forward(self, am, L1, L4):
        
        #L2 = torch.cat([self.convL2(L1), self.convAM(am)], dim=1)
        L2 = self.convL2(L1) + self.convAM(am)
        gate_L4 = self.gate(L2)
        gate_L4 = F.softmax(gate_L4, dim=1)
        
        _, c, h, w = L4.shape
        
        return (gate_L4 * L4).view(1, 2*c, h, w)
        

class SegmenterN(nn.Module):
    def __init__(self, in_ch=1):
        super(SegmenterN, self).__init__()
        self.inchannel = in_ch
        
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        
        self.convFM = nn.Conv2d(777, 256, kernel_size=(3,3), padding=(1,1), stride=1) 
        self.res_L4 = ResBlock(256, 256)        
        self.RF2 = Refine(128, 256)
        self.RF1 = Refine(in_ch+4, 256)        
        # self.pred = nn.Sequential(nn.Conv2d(256, 2, kernel_size=3, padding=1, stride=1),
        #                                     nn.BatchNorm2d(2),nn.ReLU())
        self.pred = nn.Conv2d(256, 2, kernel_size=3, padding=1, stride=1)
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, L1, L4, AM): # [1, 69, H, W], [1, 777, H/4, W/4]        
        L2 = self.layer2(L1) # 128, H/2, W/2              
        L4 = self.convFM(L4) # [b, 256, H/4, W/4]
        L4 = self.res_L4(L4) # [b, 256, H/4, W/4]
        L2 = self.RF2(L2, L4) # [b, 256, H/2, W/2]
        L1 = torch.cat([L1, AM], dim=1)
        L1 = self.RF1(L1, L2) # [b, 256, H, W]
        pred = self.pred(L1) # [b, 2, H, W]
        return pred


############################################## FEATURE EXTRACTOR
class SPX_embedding(nn.Module):
    def __init__(self, config, in_ch=3):
        super(SPX_embedding, self).__init__()
        
        if config.feat_extractor_backbone.lower() == 'resnet18':
            self.feat_extractor = ResNet18(in_ch) # img        
        
    def forward(self, img, spx):
        resft_L4, resft_L2 = self.feat_extractor(img) # L4=[b,256,w/4,h/4], L2=[b,64,w/2,h/2]
        
        # (H, W) sized feats
        resft_L1 = F.interpolate(resft_L2, (img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
        info_map = spx_info_map(spx)
        resft_L1 = torch.cat((resft_L1, info_map),1) # 67 channels
                
        # (H/4, W/4) sized feats
        _, _, h, w = resft_L4.shape
        spx_L4 = F.interpolate(spx, (h, w), mode='nearest')
        info_map_L4 = spx_info_map(spx_L4)
        resft_L4 = torch.cat((resft_L4, info_map_L4),1) # 259 channels
        
        return resft_L1, spx, resft_L4, spx_L4

############################################## MODEL
class My_Net(nn.Module):
    def __init__(self, config):
        super(My_Net, self).__init__()
        
        self.spx_embedding = SPX_embedding(config)
        self.mem_select = Mem_select()
        self.segmenter = SegmenterN(69)
        if config.fake:
            self.mem_select = Mem_select_fake()
            self.segmenter = Segmenter_fake(69)
        
        self.firstft_L1 = None
        self.lastft_L1 = None
        self.resft_L1 = None
        self.firstseg = None
        self.firstbbox = None
        self.firstimg = None
                
        self.sf_channels = config.sf_channels
                  
        self.sf_head_L1 = nn.Sequential(
            nn.Linear(67,self.sf_channels), nn.BatchNorm1d(1), nn.ReLU())
        
        self.sf_head_L4 = nn.Sequential(
            nn.Linear(259,self.sf_channels), nn.BatchNorm1d(1), nn.ReLU())
        
        self.sf_conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0), nn.ReLU())

    ##### superfeature    
    def superfeat(self, resft_L1, spx, resft_L4, spx_L4):        
        batch_size, _, _, _ = resft_L1.shape
        superft = []
        for b in range(batch_size):
            sf_L1, num_spx = self.frame_superfeat(resft_L1[b], spx[b], self.sf_head_L1)
            sf_L4, _ = self.frame_superfeat(resft_L4[b], spx_L4[b], self.sf_head_L4, num_spx)            
            fin_sf = torch.cat([sf_L1, sf_L4], dim=1)
            fin_sf = self.sf_conv_1x1(fin_sf)            
            superft.append(fin_sf[0,0])       
        return superft
    
    def frame_superfeat(self, frame_feat, frame_spx, sf_head=None, num_spx=None):
        if num_spx is None:
            num_spx = frame_spx.max().int()
        
        c, h, w = frame_feat.shape
        frame_spx = frame_spx.expand(c, h, w).int()
        frame_sf = torch.zeros((1,1,num_spx, self.sf_channels), device=frame_feat.device)
        
        for n in range(num_spx):
            if torch.count_nonzero(frame_spx[0] == n+1): 
                spx_feat = frame_feat[frame_spx==n+1].view(c,-1)                
                sf = spx_feat.mean(-1)
                frame_sf[0,0,n] = sf_head(sf[None,None])                
            else:
                frame_sf[0,0,n] = 0.0        
        return frame_sf, num_spx
    
    ##### segmentation
    # pred = model(attmaps_0, attmaps, bbox_0, bbox, lastpred, num_cls, mod='segment')
    def segment(self, attmaps_0, attmaps, bbox_0, bbox, lastpred, num_cls, r=0):        
        # bbox:  list[num_cls-1] -> [rmin, rmax, cmin, cmax]
        # attmaps:  list[num_cls-1] -> torch.Size([4, rmax-rmin, cmax-cmin])
        # lastseg:  torch.Size([b, 1, H, W])      
        # lastft_L1:  torch.Size([b, 67, H, W])
        # resft_L1:  torch.Size([b, 67, H, W]])
        
        
        # self.thisft_L1 = resft_L1
        # self.thisft_L4 = resft_L4
        # self.firstft_L1 = resft_L1.clone().detach()
        # self.firstft_L4 = resft_L4.clone().detach()
        # self.firstseg = None                
        # self.lastft_L1 = resft_L1.clone()
        # self.lastft_L4 = resft_L4.clone()
        
        
        _, _, H, W = lastpred.shape
        
        if self.firstseg is None:
            self.firstseg = lastpred.clone().float()
            self.firstbbox = bbox_0.copy()
        
        segs = []
        for c in range(1, num_cls):
            
            
            fy1, fy2, fx1, fx2 = self.firstbbox[c-1]
            fs_L1 = self.firstseg[:, c:c+1, fy1:fy2, fx1:fx2]
            ff_L1 = self.firstft_L1[r:r+1, :, fy1:fy2, fx1:fx2]
            froi = torch.tensor([[0, fx1, fy1, fx2, fy2],], dtype=ff_L1.dtype, device=ff_L1.device)
            
            ly1, ly2, lx1, lx2 = bbox_0[c-1]            
            lam = attmaps_0[c-1].unsqueeze(0)
            ls_L1 = lastpred[:, c:c+1, ly1:ly2, lx1:lx2]
            lf_L1 = self.lastft_L1[:, :, ly1:ly2, lx1:lx2]
            lroi = torch.tensor([[0, lx1, ly1, lx2, ly2],], dtype=lf_L1.dtype, device=lf_L1.device)
            
            ty1, ty2, tx1, tx2 = bbox[c-1]
            tam = attmaps[c-1].unsqueeze(0)
            tf_L1 = self.thisft_L1[:, :, ty1:ty2, tx1:tx2]
            troi = torch.tensor([[0, tx1, ty1, tx2, ty2],], dtype=tf_L1.dtype, device=tf_L1.device)
            
            fs_L1 = F.interpolate(fs_L1, (ty2-ty1,tx2-tx1), mode='nearest')
            ff_L1 = F.interpolate(ff_L1, (ty2-ty1,tx2-tx1), mode='bilinear')
            
            lam = F.interpolate(lam, (ty2-ty1,tx2-tx1), mode='bilinear')
            ls_L1 = F.interpolate(ls_L1.float(), (ty2-ty1,tx2-tx1), mode='bilinear')
            lf_L1 = F.interpolate(lf_L1, (ty2-ty1,tx2-tx1), mode='bilinear')
                        
            am = torch.cat([lam,tam], dim=0)
            f_L1 = torch.cat([lf_L1,tf_L1], dim=0)
            
            _, _, h, w = f_L1.shape
            f_L1, pad = self.pad_divide_by(f_L1, 4, (h, w))
            am, _ = self.pad_divide_by(am, 4, (h, w))
            
            
            y_L4, x_L4 = int(f_L1.shape[2]/4), int((f_L1.shape[3])/4)
            roi_align = RoIAlign((y_L4, x_L4), spatial_scale=1/4, sampling_ratio=-1)
            
            ff_L4 = roi_align(self.firstft_L4[r:r+1], froi)
            lf_L4 = roi_align(self.lastft_L4, lroi)
            tf_L4 = roi_align(self.thisft_L4, troi)
            
            f_L4 = torch.cat([lf_L4,tf_L4], dim=0)
            
            mem_L4 = self.mem_select(am, f_L1, f_L4)
            
            L4 = torch.cat([ff_L4, mem_L4], dim=1)
            
            L1 = torch.cat([fs_L1, ls_L1, ff_L1], dim=1)
            L1, _ = self.pad_divide_by(L1, 4, (h, w))
            tam_L1, _ = self.pad_divide_by(tam, 4, (h, w))
            
            seg = self.segmenter(L1, L4, tam_L1) # ch = 69, 777
            
            
            # mem_L4:  torch.Size([1, 518, 55, 61]) True
            # am_0:  torch.Size([1, 4, 157, 80]) False
            # am:  torch.Size([1, 4, 157, 80]) False
            # fs_L1:  torch.Size([1, 1, 125, 256]) False
            # ff_L1:  torch.Size([1, 67, 125, 256]) False
            # ls_L1:  torch.Size([1, 1, 125, 256]) False
            # lf_L1:  torch.Size([1, 67, 125, 256]) True
            # tf_L1:  torch.Size([1, 67, 125, 256]) True
            # ff_L4:  torch.Size([1, 259, 31, 64]) False
            # lf_L4:  torch.Size([1, 259, 31, 64]) True
            # tf_L4:  torch.Size([1, 259, 31, 64]) True
            
            
            
        #     rmin, rmax, cmin, cmax = bbox[c-1]
            
        #     ls = lastpred[:, c:c+1, rmin:rmax, cmin:cmax]
        #     lf = self.lastft_L1[:, :, rmin:rmax, cmin:cmax]
        #     rf = self.resft_L1[:, :, rmin:rmax, cmin:cmax]
        #     am = attmaps[c-1].unsqueeze(0)
        #     #li = img[:, :, rmin:rmax, cmin:cmax]
            
        #     ft = torch.cat([fs, ff, ls,lf,rf,am], dim=1)
        #     #ft = torch.cat([fi, li, fs, ff, ls,lf,rf,am], dim=1)
        #     if not ft.requires_grad: ft.requires_grad = True
        #     #ft.requires_grad = True
            
        #     _, _, h, w = ft.shape
        #     ft, pad = self.pad_divide_by(ft, 4, (h, w))
            
        #     seg = self.segmenter(ft)
            
            seg = self.pad_remove(seg, pad)
            # ty1, ty2, tx1, tx2 
            # rmin, rmax, cmin, cmax
            bground = (tx1, W-tx2, ty1, H-ty2)            
            bg_offset = torch.zeros([1, 2, H, W], device=lastpred.device)
            bg_offset[:,0,:ty1] = 1e7
            bg_offset[:,0,ty2:] = 1e7
            bg_offset[:,0,:,:tx1] = 1e7
            bg_offset[:,0,:,tx2:] = 1e7            
            segs.append(F.pad(seg, bground, "constant", 0) + bg_offset)
            # out:  torch.Size([1, 2, 256, 256]) True       
        
        out = torch.cat(segs, dim=0)            
        #torch.save(out.clone().detach().cpu(), 'dados/out_cat.pt')
        out = F.softmax(out, dim=1)[:,1] # no, h, w        
        #torch.save(out.clone().detach().cpu(), 'dados/out_softmax.pt')
        out = self.soft_aggregation(out, num_cls)
        #torch.save(out.clone().detach().cpu(), 'dados/out_agg.pt')
        #print('Tudo salvo!!!')        
        return out    
    
    
    def soft_aggregation(self, pred, num_cls):
        _, H, W = pred.shape 
        em = torch.zeros([1, num_cls, H, W], device=pred.device, dtype=pred.dtype)
        em[0,0] =  torch.prod(1-pred, dim=0) # bg_prob: [1, num_cls, H, W]
        em[0,1:num_cls] = pred # fg_prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        return torch.log((em /(1-em)))
    
    def pad_divide_by(self, x, d, in_size):
        h, w = in_size
        if h % d > 0:
            new_h = h + d - h % d
        else:
            new_h = h
        if w % d > 0:
            new_w = w + d - w % d
        else:
            new_w = w
        lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
        lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
        pad_array = (int(lw), int(uw), int(lh), int(uh))       
        return F.pad(x, pad_array), pad_array
    
    def pad_remove(self, x, pad):
        if pad[2]+pad[3] > 0:
            x = x[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            x = x[:,:,:,pad[0]:-pad[1]]
        return x
        
    
    ##### forward
    def forward(self, *args, **kwargs):
        
        # get super features: expected args = img, spx
        if 'mod' not in kwargs.keys() or 'superfeat' in kwargs.values():
            
            # resft_L1, spx, resft_L4, spx_L4
            resft_L1, spx, resft_L4, spx_L4 = self.spx_embedding(*args)
            
            self.thisft_L1 = resft_L1
            self.thisft_L4 = resft_L4
            
            if 'n' in kwargs.keys() and kwargs['n'] == 0:
                self.firstft_L1 = resft_L1.clone().detach()
                self.firstft_L4 = resft_L4.clone().detach()
                self.firstseg = None                
                self.lastft_L1 = resft_L1[:1].clone()
                self.lastft_L4 = resft_L4[:1].clone()
                
            
            # resft_L1, spx, resft_L4, spx_L4
            sf_L1 = resft_L1.clone()#.detach()
            sf_L4 = resft_L4.clone()#.detach()
            return self.superfeat(sf_L1, spx, sf_L4, spx_L4)
        
        # final segmentation: # expected args = att_maps, bbox, num_cls
        elif 'segment' in kwargs.values():
            seg = self.segment(*args)
            
            self.lastft_L1 = self.thisft_L1.clone()
            self.lastft_L4 = self.thisft_L4.clone()
            
            return seg
        
        else:
            raise RuntimeError('Invalid model arguments.')


        
####################### MEMORY ############################
class memoryClusterer(nn.Module):
    def __init__(self, config):
        super(memoryClusterer, self).__init__()
        self.cfg = config
        self.K = config.knn_neighbors
        self.knn = None 
        self.max_mem_size = None
        self.keep_init_mem = True
        self.num_cls = None
        self.min_max_n_clusters = (2, 2)#config.kmeans_max_clusters)
        #self.min_max_n_clusters = (2, config.nkc)
        self.n_clusters = []
        self.kmeans = []
        self.X_mem = []
        self.y_mem = []
        self.init_mem_size = []
        self.init_bbox = []
        self.nkc = config.nkc
        self.r = config.r
        #self.min_max_n_clusters = config.nkc
    
    def fit(self, X, y, num_cls=None):
        self.reset()                    
        self.num_cls = num_cls if num_cls else y.max().item()
        self.knn = FaissKNeighbors(k=self.K, r=self.r)
        self.knn.fit(X,y, self.num_cls)        
        for c in range(self.num_cls):
            yc = y==c+1
            nc = ((torch.count_nonzero(yc)/yc.shape[0])*self.min_max_n_clusters[1]).int().item()
            self.n_clusters.append(max(self.min_max_n_clusters[0], nc))            
            self.kmeans.append(FaissKMeans(self.n_clusters[c]))
            self.kmeans[c].fit(X[yc])
            self.X_mem.append(X[yc])
            self.y_mem.append(y[yc])
            self.init_mem_size.append(y[yc].shape[0]*5)
            #self.init_mem_size.append(y[yc].shape[0])
        self.max_mem_size = y.shape[0] * 10
        #self.max_mem_size = y.shape[0] * self.nkc + max(self.init_mem_size)
        
        
    
    def reset(self, ):
        self.__init__(self.cfg)
    
    
    def predict(self, X, spx, lastseg, lastpred, gt=None):
        # X[num_spx, sf_ch]
        # spx[b, 1, H, W] 
        # lastseg[b, 1, H, W]
        # lastpred[b, num_cls, H, W]
        # gt[b, 1, H, W]
        #lastpred = to_onehot(lastseg, self.num_cls)
        
        
        knn_pred, knn_dist = self.knn.predict(X, dist_neigh=1)        
        cent_dist = []
        last_bbox = []
        for c in range(self.num_cls):
            cent_dist.append(self.kmeans[c].predict(X))
            last_bbox.append(self.get_roi(lastseg[0,0]==c+2, r=0))
        cent_dist = torch.cat(cent_dist, dim=1)
        
        if not self.init_bbox: self.init_bbox = last_bbox.copy()
        
        # knn_pred:  torch.Size([572]) cuda:0 torch.int32
        # knn_dist:  torch.Size([572, 2]) cuda:0 torch.float32
        # cent_dist:  torch.Size([572, 2]) cuda:0 torch.float32        
        
        _, _, H, W = spx.shape
        preseg = torch.zeros([H, W], dtype=spx.dtype, device=spx.device)
        knn_att = torch.zeros([H, W, self.num_cls], dtype=X.dtype, device=spx.device)
        cent_att = torch.zeros([H, W, self.num_cls], dtype=X.dtype, device=spx.device)
        
        for i in range(spx.max()):
            s = spx[0,0]==i+1
            preseg[s] = knn_pred[i]
            knn_att[s] = knn_dist[i]
            cent_att[s] = cent_dist[i]       
        
        preseg = filt_outliers(preseg, lastseg[0,0], self.num_cls, u=0.05)        # u=0.05
        bbox = []
        attmaps = []
        
        for c in range(2, self.num_cls+1):
            
            idx_ = torch.tensor(list(range(0,c-1)) + list(range(c,self.num_cls)), device=knn_att.device)           
            knn_contrast = torch.max(torch.index_select(knn_att, 2, idx_), dim=2)[0]
            cent_contrast = torch.max(torch.index_select(cent_att, 2, idx_), dim=2)[0]                       
            
            if gt is None:
                #rmin, rmax, cmin, cmax = self.get_roi(torch.logical_or(lastseg[0,0]==c, preseg==c), r=self.nkc)
                rmin, rmax, cmin, cmax = self.get_roi(preseg==c, r=0.3) # r=0.3
                #rmin, rmax, cmin, cmax = self.get_roi(lastseg[0,0]==c, r=self.nkc)
                try:
                    rmin, rmax, cmin, cmax = self.min_bbox_size([rmin, rmax, cmin, cmax], self.init_bbox[c-2], H, W)
                except:
                    print('')
            else:                
                #rf = random.uniform(0.2, 0.5)
                rmin, rmax, cmin, cmax = self.get_roi(gt[0,0]==c)            
                
            bbox.append([rmin, rmax, cmin, cmax])                        
            
            am = torch.stack([knn_att[rmin:rmax,cmin:cmax, c-1],
                              knn_contrast[rmin:rmax,cmin:cmax],
                              cent_att[rmin:rmax,cmin:cmax, c-1],
                              cent_contrast[rmin:rmax,cmin:cmax]])            
            
            attmaps.append(am)                  
        
        return preseg[None, None], attmaps, bbox


    def min_bbox_size(self, bbox, init_bbox, H, W):           
        min_H = init_bbox[1] - init_bbox[0]
        min_W = init_bbox[3] - init_bbox[2]        
        bb_H = bbox[1] - bbox[0]
        bb_W = bbox[3] - bbox[2]        
        dif_H = bb_H - min_H
        dif_W = bb_W - min_W
        if dif_H < 0:             
            sH = int(abs(dif_H/2))
            bbox[0] = max(0, bbox[0] - sH)
            bbox[1] = min(bbox[1] + sH, H)        
        if dif_W < 0:                
            sW = int(abs(dif_W/2))
            bbox[2] = max(0, bbox[2] - sW)
            bbox[3] = min(bbox[3] + sW, W)
        return bbox
            

    def get_roi(self, mask, r=0.5, m=0.05):
        H, W = mask.shape        
        
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        try:
            rmin, rmax = torch.where(rows)[0][[0, -1]]
            cmin, cmax = torch.where(cols)[0][[0, -1]]
            rmin, rmax, cmin, cmax = rmin.item(), rmax.item()+1, cmin.item(), cmax.item()+1
        except:
            rmin, rmax, cmin, cmax = 0, H, 0, W
            return (rmin, rmax, cmin, cmax)        
        radd = int((rmax-rmin)*r/2)
        cadd = int((cmax-cmin)*r/2)
        rmin = max(0, rmin - radd)
        rmax = min(H, rmax + radd)
        cmin = max(0, cmin - cadd)
        cmax = min(W, cmax + cadd)        
        if rmax - rmin < m*H:
            rmin, rmax = 0, H
        if cmax - cmin < m*W:
            cmin, cmax = 0, W
        return (rmin, rmax, cmin, cmax)
    
    def update(self, X, y):        
        for c in range(self.num_cls):
            yc = y==c+1        
            new_mem_size = self.X_mem[c].shape[0] + X[yc].shape[0]
            idx = [*range(new_mem_size)] 
            if (self.max_mem_size is not None) and (new_mem_size > self.max_mem_size):
                if self.keep_init_mem:
                    idx = [*range(self.init_mem_size[c]),
                           *range(self.init_mem_size[c]+new_mem_size-self.max_mem_size, new_mem_size)]
                else:
                    idx = [*range(new_mem_size-self.max_mem_size, new_mem_size)]         
            self.X_mem[c] = torch.cat([self.X_mem[c], X[yc]], dim=0)[idx]
            self.y_mem[c] = torch.cat([self.y_mem[c], y[yc]], dim=0)[idx]            
            self.kmeans[c].fit(self.X_mem[c], init_centroids=self.kmeans[c].centroids)            
        self.knn.fit(torch.cat(self.X_mem, dim=0), torch.cat(self.y_mem, dim=0), self.num_cls)
    

####################### LOSS ##############################
class My_Loss(nn.Module):
    def __init__(self, config):
        super(My_Loss, self).__init__()
        self.bypass_loss = config.bypass_loss 
        
        # metric learning
        # reducer
        if config.reducer == 'PerAnchorReducer':
            self.reducer = reducers.PerAnchorReducer()
        elif config.reducer == 'AvgNonZeroReducer':
            self.reducer = reducers.AvgNonZeroReducer()
        else:            
            self.reducer = None
        # npair loss
        if config.loss_ML == 'compose':
            npair_loss = losses.NTXentLoss(temperature=0.07, reducer=self.reducer)
            var_loss = losses.IntraPairVarianceLoss(distance=distances.CosineSimilarity())
            self.loss_ML_fun = losses.MultipleLosses([npair_loss, var_loss], weights=[1, 1])
        else:
            self.loss_ML_fun = losses.NTXentLoss(temperature=0.07, reducer=self.reducer)
        # miner
        if config.miner == 'BatchEasyHardMiner':
            self.miner = miners.BatchEasyHardMiner(pos_strategy='easy', neg_strategy='semihard',
                                                   allowed_pos_range=None,allowed_neg_range=None,
                                                   distance=distances.CosineSimilarity())
        elif config.miner == 'UniformHistogramMiner':
            self.miner = miners.UniformHistogramMiner(num_bins=100, pos_per_bin=10, neg_per_bin=10,
                                                      distance=distances.CosineSimilarity())
        elif config.miner == 'BatchHardMiner':
            self.miner = miners.BatchHardMiner(distance=distances.CosineSimilarity())
        else:
            self.miner = None
        self.t_per_anchor = config.t_per_anchor
        
        # cross entropy loss
        self.loss_CE_fun = nn.CrossEntropyLoss()
    
    def metric_learning(self, spx_pools, super_feat):
        loss = torch.tensor([0.0], device=super_feat[0].device)
        if 'metric_learning' in self.bypass_loss:
            return loss
        
        batch_size = len(spx_pools)
        for b in range(batch_size):            
            if super_feat[b].shape[0] == spx_pools[b].shape[0]:
                if self.miner is not None:
                    indices_tuple = self.miner(super_feat[b], spx_pools[b])
                else:
                    indices_tuple = lmu.get_random_triplet_indices(
                        spx_pools[b], t_per_anchor=self.t_per_anchor)
                
                loss += self.loss_ML_fun(super_feat[b], spx_pools[b], indices_tuple)            
            
        return loss/batch_size
    
    def cross_entropy(self, pred, label):
        if 'cross_entropy' in self.bypass_loss:
            return torch.tensor([0.0], device=pred.device)        
        
        ce_loss = self.loss_CE_fun(pred,label)        
        
        return ce_loss
    
    def forward(self, spx_pools=None, super_feat=None, pred=None, label=None):
        
        if spx_pools is not None and super_feat is not None:
            return self.metric_learning(spx_pools, super_feat)
        
        elif pred is not None and label is not None:
            return self.cross_entropy(pred, label)
        else:
            raise RuntimeError('Invalid loss arguments.')


####################################################################
def get_model_loss(config, rank):    
    model = My_Net(config)
    loss_fun = My_Loss(config)
    
    frozen = []
    for fm in config.freeze_modules:        
        for (name, module) in model.named_children():
            if name.find(fm) != -1:
            #if name == config.modules[fm]:
                frozen.append(name)
                for layer in module.children():
                    for param in layer.parameters():
                        param.requires_grad = False
    if frozen and rank==0: print('Frozen modules: {}'.format(frozen))
    
    return model, loss_fun

def get_memory(config):    
    return memoryClusterer(config)


def get_opti_scheduler(config, model):
    optimizer = None
    lr_scheduler = None
    if config.classifier_optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                     lr=config.learning_rate)        
    if config.classifier_optimizer == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True)        
    if config.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
    return optimizer, lr_scheduler