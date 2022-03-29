#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os.path
import time
from torch.nn.parallel import DistributedDataParallel as DDP
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

def launch_cuda_ddp(model, rank):
    global device
    device=torch.device("cuda:{}".format(rank))
    if rank == 0:
        print('CUDA version {} [{} device(s) available].'.format(torch.version.cuda, torch.cuda.device_count()))
    model = model.to(rank)    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
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
    # set next epoch to resume the training
    start_epoch = checkpoint['epoch'] + 1
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
        return model, optimizer, start_epoch
    
    return model, start_epoch

def save_model(config, model, model_name, epoch, optimizer=None):
    
    if os.path.exists(config.save_model_path) is False:
        os.makedirs(config.save_model_path)
    
    path_ = os.path.join(config.save_model_path, config.model + model_name)
    
    if optimizer is not None:    
        torch.save({'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,}, path_)
    else:
        torch.save({'model': model.state_dict(),
                    'epoch': epoch,}, path_)
    
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

def merge_spx_label(spx, obj_label, spx2label=None):
    
    if spx2label is None:
        spx2label = obj_label.clone()
    
    num_spx = spx.max().item()
    output = torch.zeros_like(spx, device=device)
    
    for i in range(1, obj_label.max()+1):        
        n_spx = spx.clone()
        n_spx[spx2label != i] = 0
        
        n_obj_label = obj_label.clone() + num_spx
        n_obj_label[obj_label != i] = 0
        
        r_spx = n_spx + n_obj_label        
        x_spx = r_spx.clone()        
        x_spx[r_spx != num_spx+i] = 0
        r_spx = r_spx - num_spx - i
        r_spx[obj_label != i] = 0
        r_spx[x_spx == num_spx+i] = 0
        r_spx = r_spx + x_spx
        
        output = output + r_spx
        
    # reorder spx labels
    all_labels = torch.tensor(range(1, output.max()+1))
    present_labels = all_labels.clone().apply_(lambda x: x in output)    
    gap = 0
    cor = 0
    for i, k in enumerate(all_labels):            
        if present_labels[i] == 0:
            gap += 1
            cor += 1
        else:
            output[output>=k-cor] -= gap
            gap = 0
    
    return output


def iou_metrics(obj_label, spx, pred, y_train, idx_train, idx_test):
    train_spx = torch.zeros_like(spx, device=device)
    test_spx = torch.zeros_like(spx, device=device)
    
    for n, i in enumerate(idx_train):    
        train_spx[spx==i] = y_train[n]
    
    for n, i in enumerate(idx_test):    
        test_spx[spx==i] = pred[n]
    
    i = torch.count_nonzero(obj_label[train_spx==0] == test_spx[train_spx==0])    
    u = torch.count_nonzero(obj_label[train_spx==0] + test_spx[train_spx==0])
    iou = i/u if u>0 else 0.0
    
    obj_label[train_spx>0] = 0
    test_spx[train_spx>0] = 0
    
    iiou = 0.0
    for k in range (1, obj_label.max()+1):
        ii = torch.count_nonzero((obj_label==k).int() * (test_spx==k).int())
        uu = torch.count_nonzero((obj_label==k).int() + (test_spx==k).int())
        iiou = (iiou + ii/uu) if uu>0 else iiou
    iiou /= k 
    
    return iou, iiou, (iou+iiou)/2

def show_intro(config, delay=0.1, size=20):
    print('\n[{}: {}] >\b'.format(config.model, config.experiment_name), end="", flush=True)
    for i in range(size):
        print('= >\b', end="", flush=True)
        time.sleep(delay)
    print('> [started]')

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
