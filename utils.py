#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os.path
import time
import numpy as np
from skimage.measure import label
from torch.nn.parallel import DistributedDataParallel as DDP
#import torch.nn as nn
#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

def launch_cuda_ddp(model, rank, optimizer=None):
    global device
    device=torch.device("cuda:{}".format(rank))
    if rank == 0:
        print('CUDA version {} [{} device(s) available].'.format(torch.version.cuda, torch.cuda.device_count()))
    model = model.to(rank)    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    if optimizer is not None:
        optimizer = optimizer_ddp(optimizer,device) 
        return model, optimizer
    
    return model

def optimizer_ddp(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optim

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


def load_checkpoint(config, model, optimizer=None, rank=None, train=False):
    # load saved model if specified
    load_name = os.path.join(config.resume_model_path)    
    if not os.path.isfile(load_name):
        raise RuntimeError('File not found: {}'.format(load_name))       
    
    if rank is not None:
        checkpoint = torch.load(load_name, map_location=torch.device("cuda:{}".format(rank)))
    else:
        checkpoint = torch.load(load_name) # dict_keys(['epoch', 'model', 'optimizer'])
    
    start_epoch = checkpoint['epoch'] + 1
    try:        
        top_iou = checkpoint['top_iou']
        global_ite = checkpoint['global_ite']
        test_ite = checkpoint['test_ite']
    except:
        top_iou, global_ite, test_ite = 0, 0, 0
    
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model']
    pretrained_dict_update = {}
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
    
    model_dict.update(pretrained_dict_update)
    model.load_state_dict(model_dict)    

    if rank is None or rank==0:
        print('Loaded checkpoing at: {}'.format(load_name))
    
    if optimizer is not None:
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        del(checkpoint)
        if train:
            return model, optimizer, start_epoch, top_iou, global_ite, test_ite
        else:
            return model, optimizer
    
    del(checkpoint)
    if train:
        return model, start_epoch, top_iou, global_ite, test_ite
    else:
        return model

def save_model(config, model, model_name, epoch=0, optimizer=None, top_iou=0.0, global_ite=0, test_ite=0):
    
    if os.path.exists(config.save_model_path) is False:
        os.makedirs(config.save_model_path)
    
    path_ = os.path.join(config.save_model_path, config.model + model_name)
    
    if optimizer is not None:    
        torch.save({'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'top_iou': top_iou,
                    'global_ite': global_ite,
                    'test_ite': test_ite,}, path_)
    else:
        torch.save({'model': model.state_dict(),
                    'epoch': epoch,
                    'top_iou': top_iou,
                    'global_ite': global_ite,
                    'test_ite': test_ite,}, path_)
    
    print('Model saved in {}'.format(path_))

def get_spx_pools(spx, obj_labels):
# output: spx per obj (greater intersection = statistic mode)
# [spx, obj_label]
    batch_size, _, _, _ = spx.shape
    pools = []
    new_spx = torch.zeros_like(spx, device=device)
        
    for b in range(batch_size):
    
        num_spx = spx[b,0].max().item()        
        p = torch.zeros(num_spx, dtype=new_spx.dtype, device=device)
        
        for s in range(num_spx):

            p[s] = obj_labels[b,0,spx[b,0]==s+1].mode()[0]

            new_spx[b,0,spx[b,0]==s+1] = p[s]
        
        # remove unique elements
        for x in torch.where(torch.bincount(p)==1)[0]:
            p[p==x] = 1
            new_spx[b,0,new_spx[b,0]==x] = 1
        
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
    # info_map = [b,4,w,h] -> ch0=spx_size, ch1=spx_xmap, ch2=spx_ymap, ch3=spx_idx_normalized
    labels = labels.int()
    b, c, w, h = labels.shape
    info_map = torch.zeros([b,3,w,h], device=device)
    
    xvalues = torch.arange(0,w, device=device)
    yvalues = torch.arange(0,h, device=device)
    xx, yy = torch.meshgrid(xvalues/(w-1), yvalues/(h-1))   
    
    for b_idx in range(b):
        label_max = labels[b_idx].max()
        for n in range(1, label_max+1):
            spx_size = torch.nonzero(labels[b_idx]==n).shape[0] / (w*h)
            info_map[b_idx,0,labels[b_idx,0]==n] = spx_size            
            info_map[b_idx,1,labels[b_idx,0]==n] = xx[labels[b_idx,0]==n].mean()
            info_map[b_idx,2,labels[b_idx,0]==n] = yy[labels[b_idx,0]==n].mean()
        #info_map[b_idx,3] = labels[b_idx] / label_max
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
        z_spx = torch.from_numpy(label((x_spx>0).cpu().numpy().astype(np.int32), 
                                       connectivity=1, return_num=False)).to(device)        
        r_spx = r_spx + x_spx + z_spx
        
        output = output + r_spx
        
        num_spx = output.max().item()
    
    return reoerder_labels(output)


def reoerder_labels(labels, first_label=1):
    # reorder the sequence of labels by removing inexistent values
    all_labels = torch.tensor(range(first_label, labels.max()+1))
    present_labels = all_labels.clone().apply_(lambda x: x in labels)    
    gap = 0
    cor = 0
    for i, k in enumerate(all_labels):            
        if present_labels[i] == 0:
            gap += 1
            cor += 1
        else:
            labels[labels>=k-cor] -= gap
            gap = 0
    
    return labels


def iou_metrics(obj_label, spx, pred, y_train, idx_train, idx_test):
    train_spx = torch.zeros_like(spx, device=device)
    test_spx = torch.zeros_like(spx, device=device)
    
    for n, i in enumerate(idx_train):    
        train_spx[spx==i] = y_train[n]
    
    for n, i in enumerate(idx_test):    
        test_spx[spx==i] = pred[n]
    
    iou = 0.0
    cc = 0
    valid = (train_spx==0).int()
    for k in range(2, obj_label.max()+1):
        
        ground_truth = (obj_label==k).int()
        prediction = (test_spx==k).int()
        
        i_ = torch.count_nonzero(ground_truth * prediction * valid)
        u_ = torch.count_nonzero((ground_truth + prediction) * valid)
        
        iou += i_/u_ if u_>0 else 0.0
        cc += 1
    if cc > 0:
        iou = iou/cc
    
    return iou    
    

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

#plt.ion()
def plot_grad_flow(named_parameters):    
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
            else:
                ave_grads.append(0.0)      
    plt.figure(num=1) 
    plt.plot(ave_grads, alpha=0.8, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.draw()
    plt.savefig('grad_plot_bad.png')  
    plt.show() 
    plt.close()


def plot_grad_flow_2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
                #print('n: {}, grad_max: {}, grad_mean: {}, grad_fn: {}'.format(n, p.grad.abs().max(), p.grad.abs().mean(), p.grad_fn))
            else:
                ave_grads.append(0.0)
                max_grads.append(0.0)
                #print('n: {}, grad: None'.format(n))
        
    plt.figure(num=2) 
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.8, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.8, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('grad_plot_2_bad.png')  
    plt.show() 
    plt.close()
    