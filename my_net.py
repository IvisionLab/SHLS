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
from utils import launch_cuda, get_spx_pools, AverageMeter, iou_metrics 
from utils import load_checkpoint, save_model, ungroup_batches, show_intro


def train(config, model, train_loader, test_loader, loss_function, optimizer, lr_scheduler):
    global_ite = 0
    start_epoch = 0
    test_ite = 0
    writer = SummaryWriter(config.save_model_path)
    
    model = launch_cuda(model)
    loss_function = loss_function.cuda()
        
    if config.resume_model_path:
        model, optimizer, start_epoch, global_ite = load_checkpoint(config, model, optimizer)

    for e in range(start_epoch, config.epoch):
        train_loss = AverageMeter()
        train_dataiter = iter(train_loader)
        t = tqdm(train_dataiter)
        t.set_description("Train epoch [{}/{}]".format(e+1,config.epoch))
        
        for i, (img, spx, obj_label, _, _) in enumerate(t):
            global_ite += 1

            # reset model's mode
            model.train()

            # input image, generated superpixels and object pseudo-labels
            img = img.float().cuda()
            spx = spx.cuda()
            obj_label = obj_label.cuda()
            
            # pools of superpixels by object
            spx_pools, _ = get_spx_pools(spx, obj_label)
            
            # feature embedding by superpixel
            #*spx_emb, = model.spx_embedding(img, spx.float())
            
            # make super features from superpixels and embeddings
            #super_feat = model.super_feat(*spx_emb)
            
            super_feat = model(img, spx.float())
            
            # if necessary, ungroup and reorganize feat batches from different gpus
            spx_pools, super_feat = ungroup_batches(spx_pools, super_feat)
            
            # compute loss with metric learning
            loss = compute_loss(loss_function, spx_pools, super_feat, config.t_per_anchor)

            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get numerical loss value
            #loss_value = loss.cpu().detach().data.numpy()
            train_loss.update(loss.item())

            # test stage
            if global_ite == 1: # % config.test_times == 0:
                test_ite = test(config, model, test_loader, loss_function, e+1, writer, test_ite)
            
            # save model
            if global_ite % config.save_model_times == 0:
                model_name = '_epoch_'+str(e)+'_it_'+str(global_ite)+'.pth'
                save_model(config, model, model_name, e, global_ite, optimizer)
            
            # write the summary
            #writer.add_scalar('Train: loss_value', train_loss.val, global_ite)
            writer.add_scalar('Train: loss_avg', train_loss.avg, global_ite)
            
            t.set_postfix_str('loss: {:^7.3f} (Avg: {:^7.3f})'.format(train_loss.val, train_loss.avg))
            t.update()
          
        test_ite = test(config, model, test_loader, loss_function, e+1, writer, test_ite)
        save_model(config, model, '_last.pth', e, global_ite, optimizer)
        print("Finished epoch [{}/{}]".format(e+1,config.epoch))

def test(config, model, test_loader, loss_function, epoch, writer, test_ite):
    model.eval()
    
    with torch.no_grad():                
        test_loss = AverageMeter()
        knn_score = AverageMeter()
        iou = AverageMeter()
        iiou = AverageMeter()
        _2iou_mean = AverageMeter()
        test_dataiter = iter(test_loader)
        t = tqdm(test_dataiter)        
        knn = KNeighborsClassifier(n_neighbors = config.knn_neighbors)
        t.set_description("Test [epoch={}]".format(epoch))
        skipped_samples = 0
        
        for i, (img, spx, obj_label, num_obj, info) in enumerate(t):
            test_ite += 1
            
            img = img.float().cuda()
            spx = spx.cuda()
            obj_label = obj_label.cuda()
            
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
                    _iou, _iiou, _2iou = iou_metrics(obj_label[b,0].clone(), spx[b,0].clone(), pred, y_train, idx_train, idx_test)
                    iou.update(_iou), iiou.update(_iiou), _2iou_mean.update(_2iou)
                except:
                    skipped_samples += 1
                    continue
            
            writer.add_scalar('Test: loss_avg', test_loss.avg, test_ite)
            writer.add_scalar('Test: knn_score', knn_score.avg, test_ite)
            writer.add_scalar('Test: IoU', iou.avg, test_ite)
            writer.add_scalar('Test: iIoU', iiou.avg, test_ite)
            writer.add_scalar('Test: Mean(IoU, iIoU)', _2iou_mean.avg, test_ite)
            
            t.set_postfix_str('loss:{:^7.3f}knn:{:^7.3f}IoU:{:^7.2f}iIoU:{:^7.2f}({:^7.2f})'.format(
                test_loss.avg, knn_score.avg, iou.avg, iiou.avg, _2iou_mean.avg))
            t.update()
    
    print('Knn: {}/{} samples skipped at test.'.format(skipped_samples, i+1))
    
    return test_ite


def run(config):
    # data 
    train_loader, test_loader   = _Data.get_data(config)
    # model & loss
    model, loss_function        = _Models.get_model_loss(config)
    # optimizer & scheduler
    optimizer, lr_scheduler     = _Models.get_opti_scheduler(config, model, train_loader)
    # training & online testing
    if True:
        train(config, model, train_loader, test_loader, loss_function, optimizer, lr_scheduler)


if __name__ == '__main__':
    config = set_config()    
    show_intro(config)    
    run(config)