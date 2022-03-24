#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from config import set_config
import Data.get_data as _Data
import Models.get_model as _Models
from Models.get_model import compute_loss
from utils import launch_cuda_ddp, get_spx_pools, AverageMeter, iou_metrics 
from utils import load_checkpoint, save_model, ungroup_batches, show_intro
import torch.distributed as dist
import os
import torch.multiprocessing as mp

def train(config, model, train_loader, test_loader, loss_function, optimizer, lr_scheduler, rank=None, world_size=None):
    global_ite = 0
    start_epoch = 0
    test_ite = 0
    top_iou = 0.0
    
    model = launch_cuda_ddp(model, rank)
    loss_function = loss_function.cuda(rank)
        
    if config.resume_model_path:
        model, optimizer, start_epoch = load_checkpoint(config, model, optimizer)
        
    if rank == 0:
        writer = SummaryWriter(config.save_model_path)
    else:
        writer = None

    for e in range(start_epoch, config.epoch):
        train_loader.sampler.set_epoch(e)
        train_loss = AverageMeter()
        train_dataiter = iter(train_loader)
        t = train_dataiter
        if rank == 0:
            t = tqdm(t)
            t.set_description("Train epoch [{}/{}]".format(e+1,config.epoch))
        
        for i, (img, spx, obj_label, _, _) in enumerate(t):
            global_ite += 1

            # reset model's mode
            model.train()

            # input image, generated superpixels and object pseudo-labels
            img = img.float().cuda(rank)
            spx = spx.cuda(rank)
            obj_label = obj_label.cuda(rank)
            
            # pools of superpixels by object
            spx_pools, _ = get_spx_pools(spx, obj_label)
            
            # make super features from superpixels and embeddings
            super_feat = model(img, spx.float())
            
            # if necessary, ungroup and reorganize feat batches from different gpus
            spx_pools, super_feat = ungroup_batches(spx_pools, super_feat)
            
            # compute loss with metric learning
            loss = compute_loss(loss_function, spx_pools, super_feat, config.t_per_anchor)

            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # gather distributed losses at gpu with rank 0
            loss_outputs = [None for _ in range(world_size)]
            dist.all_gather_object(loss_outputs, loss.item())
            if rank == 0:
                for l in loss_outputs:
                    train_loss.update(l)

            # scheduled test
            if global_ite == 1: # % config.test_times == 0:
                test_ite, _ = test(config, model, test_loader, loss_function, e, writer, test_ite, rank, world_size)
            
            # scheduled checkpoint saving
            if global_ite % config.save_model_times == 0:
                model_name = '_epoch_'+str(e)+'_it_'+str(global_ite)+'.pth'
                save_model(config, model, model_name, e, global_ite, optimizer)
            
            # write the summary
            if rank == 0:
                #writer.add_scalar('Train: loss_value', train_loss.val, global_ite)
                writer.add_scalar('Train: loss_avg', train_loss.avg, global_ite)                
                t.set_postfix_str('loss: {:^7.3f} (Avg: {:^7.3f})'.format(train_loss.val, train_loss.avg))
                t.update()
        
        # test, update learning rate and save at the end of the epoch
        test_ite, iou = test(config, model, test_loader, loss_function, e+1, writer, test_ite, rank, world_size)
        if lr_scheduler is not None: lr_scheduler.step(iou) 
        if rank == 0:
            if iou > top_iou:
                top_iou = iou
                save_model(config, model, '_best.pth', e, optimizer)
            save_model(config, model, '_last.pth', e, optimizer)
            print("Finished epoch [{}/{}]".format(e+1,config.epoch))

def test(config, model, test_loader, loss_function, epoch, writer, test_ite, rank=None, world_size=None):
    model.eval()
    
    with torch.no_grad():                
        test_loss = AverageMeter()
        knn_score = AverageMeter()
        iou = AverageMeter()
        iiou = AverageMeter()
        ag_iou = AverageMeter()
        knn = KNeighborsClassifier(n_neighbors = config.knn_neighbors)
        skipped_samples = 0
        
        test_loader.sampler.set_epoch(epoch)
        test_dataiter = iter(test_loader)
        t = test_dataiter
        if rank == 0:
            t = tqdm(t)        
            t.set_description("Test [epoch={}]".format(epoch))
        
        for i, (img, spx, obj_label, num_obj, info) in enumerate(t):
            test_ite += 1
            
            img = img.float().cuda(rank)
            spx = spx.cuda(rank)
            obj_label = obj_label.cuda(rank)
            
            spx_pools, _ = get_spx_pools(spx, obj_label)
            super_feat = model(img, spx.float())
            
            spx_pools, super_feat = ungroup_batches(spx_pools, super_feat)
            
            loss = compute_loss(loss_function, spx_pools, super_feat, config.t_per_anchor)
            test_loss.update(loss.item())
            
            for b in range(len(super_feat)):            
                x = super_feat[b].clone().detach().cpu().numpy()
                y = spx_pools[b].clone().detach().cpu().numpy()
                idx = np.arange(1, y.shape[0]+1)
                
                try:                
                    x_train, x_test, y_train, y_test, idx_train, idx_test = \
                        train_test_split(x, y, idx, test_size=config.knn_test_size, stratify=y)                    
                    knn.fit(x_train,y_train)
                    pred = knn.predict(x_test)            
                    knn_score.update(knn.score(x_test, y_test))
                    _iou, _iiou, _ag_iou = iou_metrics(obj_label[b,0].clone(), spx[b,0].clone(), pred, y_train, idx_train, idx_test)
                    iou.update(_iou.item()), iiou.update(_iiou.item()), ag_iou.update(_ag_iou.item())
                except:
                    skipped_samples += 1
                    continue
            
            loss_outputs = [None for _ in range(world_size)]
            knn_outputs = [None for _ in range(world_size)]
            iou_outputs = [None for _ in range(world_size)]
            iiou_outputs = [None for _ in range(world_size)]
            ag_iou_outputs = [None for _ in range(world_size)]
            skipped_samples_outputs = [None for _ in range(world_size)]
            i_outputs = [None for _ in range(world_size)]
            
            dist.all_gather_object(loss_outputs, test_loss.avg)
            dist.all_gather_object(knn_outputs, knn_score.avg)
            dist.all_gather_object(iou_outputs, iou.avg)
            dist.all_gather_object(iiou_outputs, iiou.avg)
            dist.all_gather_object(ag_iou_outputs,  ag_iou.avg)
            dist.all_gather_object(skipped_samples_outputs, skipped_samples)
            dist.all_gather_object(i_outputs, i+1)
            
            if rank == 0:
                all_test_loss = np.mean(loss_outputs)
                all_knn_score = np.mean(knn_outputs)
                all_iou = np.mean(iou_outputs)
                all_iiou = np.mean(iiou_outputs)
                all_ag_iou = np.mean(ag_iou_outputs)
                all_skipped_samples = np.sum(skipped_samples_outputs)
                all_i = np.sum(i_outputs)                
            
                writer.add_scalar('Test: loss_avg', all_test_loss, test_ite)
                writer.add_scalar('Test: knn_score', all_knn_score, test_ite)
                writer.add_scalar('Test: IoU', all_iou, test_ite)
                writer.add_scalar('Test: iIoU', all_iiou, test_ite)
                writer.add_scalar('Test: Mean(IoU, iIoU)', all_ag_iou, test_ite)
                
                t.set_postfix_str('loss:{:^7.3f}knn:{:^7.3f}IoU:{:^7.2f}iIoU:{:^7.2f}({:^7.2f})'.format(
                    all_test_loss, all_knn_score, all_iou, all_iiou, all_ag_iou))
                t.update()
            else:
                #all_ag_iou = 0.0
                all_ag_iou = np.mean(ag_iou_outputs)
    
    if rank == 0:
        print('Knn: {}/{} samples skipped at test.'.format(all_skipped_samples, all_i))
    
    return test_ite, all_ag_iou


def run(rank, config):
    
    world_size = config.num_devices
    setup(rank, world_size)
    
    if rank == 0:
        print('Using distributed dataparallel (world_size={}).'.format(world_size))
        
    # data 
    train_loader, test_loader   = _Data.get_data_ddp(config, rank, world_size)
    # model & loss
    model, loss_function        = _Models.get_model_loss(config)
    # optimizer & scheduler
    optimizer, lr_scheduler     = _Models.get_opti_scheduler(config, model, train_loader)
    # training & online testing
    if True:
        train(config, model, train_loader, test_loader, loss_function, optimizer, lr_scheduler, rank, world_size)
   

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


if __name__ == '__main__':
    config = set_config()    
    show_intro(config)
    
    try:
        mp.spawn(run, nprocs=config.num_devices, args=(config,))
    except:
        raise RuntimeError('Unable to start Distributed Dataparallel (DDP) processes.')

    
    
    
    

