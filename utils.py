#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os.path
#import torch.nn as nn
#import numpy as np


device = torch.device("cpu")

def launch_cuda(model=None):    
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_devices = torch.cuda.device_count()
        print('CUDA version {} [{} device(s) available].'.format(torch.version.cuda, num_devices))
        if model is not None:
            model = model.to(device)
            print('Model sent to cuda.')
            if num_devices > 1:
                #model = nn.DataParallel(model)
                model = MyDataParallel(model)
                print("Model parallelised in {} GPUs".format(num_devices))    
            return model
    else:
        device = torch.device("cpu")
        print('CUDA not available, CPU will be used.')
        if model is not None:
            return model

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
#     def __setattr__(self, name, value):
#         try:
#             return super().__setattr__(name, value)
#         except AttributeError:
#             return setattr(self.module, name, value)

def load_checkpoint(config, model, optimizer=None):
    # load saved model if specified
    
    load_name = os.path.join(config.resume_model_path)
    
    if not os.path.isfile(load_name):
        raise RuntimeError('File not found: {}'.format(load_name))
      
    # get the state dict of current model 
    state = model.state_dict()
    # load entire saved model from checkpoint
    checkpoint = torch.load(load_name) # dict_keys(['epoch', 'model', 'optimizer'])
    # filter out unnecessary keys from checkpoint
    checkpoint['model'] = {k:v for k,v in checkpoint['model'].items() if k in state}
    # overwrite entries in the existing state dict
    state.update(checkpoint['model'])
    # load the new state dict
    model.load_state_dict(state)
    print('Loaded checkpoing at: {}'.format(load_name))
    # load optimizer state dict
    if optimizer is not None:
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
    
    return model

def save_model(config, model, model_name, optimizer=None):
    
    if os.path.exists(config.save_model_path) is False:
        os.makedirs(config.save_model_path)
    
    path_ = os.path.join(config.save_model_path, config.model + model_name)
    
    if optimizer is not None:    
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),}, path_)
    else:
        torch.save({'model': model.state_dict(),}, path_)
    
    print('Model saved in {}'.format(path_))

def get_spx_pools(spx, obj_labels):
# output: spx by obj (greater intersection)
# [spx, obj_label]
    batch_size, _, _, _ = spx.shape
    pools = []
    new_spx = torch.zeros_like(spx, device=device)
        
    for b in range(batch_size):
    
        num_spx = spx[b,0].max().item()        
        p = torch.zeros(num_spx, dtype=new_spx.dtype, device=device)
        
        for s in range(num_spx):

            p[s] = obj_labels[b,0][spx[b,0]==s+1].mode()[0]

            new_spx[b,0][spx[b,0]==s+1] = p[s]
        
        # remove unique elements
        for x in torch.where(torch.bincount(p)==1)[0]:
            p[p==x] = 1
            new_spx[b,0][new_spx[b,0]==x] = 1
        
        pools.append(p)
    
    return pools, new_spx

def ungroup_batches(spx_pools, super_feat):
    
    split_size = len(super_feat)
    if split_size == len(spx_pools):
        return spx_pools, super_feat
    
    pools_sizes = [x.shape[0] for x in spx_pools]
    ungrouped_sf = []
    reordered_sp = []
    
    for i in range(split_size):
        sf_splits = torch.split(super_feat[i], pools_sizes[i::split_size])
        
        for n, sf in enumerate(sf_splits):
            ungrouped_sf.append(sf)
            reordered_sp.append(spx_pools[i+n*split_size])
    
    return reordered_sp, ungrouped_sf

def spx_info_map(labels):
    # info_map = [b,3,w,h] -> ch0=spx_size, ch1=spx_xmap, ch2=spx_ymap
    labels = labels.int()
    b, c, w, h = labels.shape
    info_map = torch.zeros([b,3,w,h], device=device)
    
    xvalues = torch.arange(0,w, device=device)
    yvalues = torch.arange(0,h, device=device)
    xx, yy = torch.meshgrid(xvalues/(w-1), yvalues/(h-1))   
    
    for b_idx in range(b):
        for n in range(1, labels[b_idx].max()+1):
            spx_size = torch.nonzero(labels[b_idx]==n).shape[0] / (w*h)
            info_map[b_idx,0][labels[b_idx,0] == n] = spx_size            
            info_map[b_idx,1][labels[b_idx,0] == n] = xx[labels[b_idx,0] == n].mean()
            info_map[b_idx,2][labels[b_idx,0] == n] = yy[labels[b_idx,0] == n].mean()
            
    return info_map


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        