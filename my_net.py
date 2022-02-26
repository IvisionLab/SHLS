#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from config import set_config
import Data.get_data as _Data
import Models.get_model as _Models
from Models.get_model import compute_loss
from utils import launch_cuda, get_spx_pools, AverageMeter, load_checkpoint, save_model, ungroup_batches
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def train(config, model, train_loader, test_loader, loss_function, optimizer, lr_scheduler):
    global_ite = 0
    
    model = launch_cuda(model)
    loss_function = loss_function.cuda()
        
    if config.resume_model_path:
        model, optimizer = load_checkpoint(config, model, optimizer)

    for e in range(config.epoch):
        train_loss = AverageMeter()
        train_dataiter = iter(train_loader)
        t = tqdm(train_dataiter)
        t.set_description("Train epoch [{}/{}]".format(e+1,config.epoch))
        
        for i, (img, spx, obj_label, _, _) in enumerate(t):
            global_ite = global_ite + 1

            # reset model's mode
            model.train()

            # input image, generated superpixels and object pseudo-labels
            img = img.float().cuda()
            spx = spx.cuda()
            obj_label = obj_label.cuda()
            
            # pools of superpixels by object
            spx_pools, _ = get_spx_pools(spx, obj_label)
            
            # feature embedding by superpixel
            *spx_emb, = model.spx_embedding(img, spx.float())
            
            # make super features from superpixels and embeddings
            super_feat = model.super_feat(*spx_emb)
            
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
            if global_ite % config.test_times == 0:
                test(config, model, test_loader, loss_function, global_ite)
            
            # save model
            if global_ite % config.save_model_times == 0:
                model_name = '_epoch_'+str(e)+'_it_'+str(global_ite)+'.pth'
                save_model(config, model, model_name, optimizer)
            
            t.set_postfix_str('loss: {:^7.3f} (Avg: {:^7.3f})'.format(train_loss.val, train_loss.avg))
            t.update()
            
        save_model(config, model, '_last.pth', optimizer)
        print("Finished epoch [{}/{}]".format(e+1,config.epoch))

def test(config, model, test_loader, loss_function, global_ite):
    model.eval()
    
    with torch.no_grad():
                
        test_loss = AverageMeter()
        knn_score = AverageMeter()
        test_dataiter = iter(test_loader)
        t = tqdm(test_dataiter)        
        knn = KNeighborsClassifier(n_neighbors = config.knn_neighbors)
        t.set_description("Test [it={}]".format(global_ite))
        
        for i, (img, spx, obj_label, num_obj, info) in enumerate(t):
            
            img = img.float().cuda()
            spx = spx.cuda()
            obj_label = obj_label.cuda()
            
            spx_pools, _ = get_spx_pools(spx, obj_label)
            *spx_emb, = model.spx_embedding(img, spx.float())
            super_feat = model.super_feat(*spx_emb)
            
            loss = compute_loss(loss_function, spx_pools, super_feat, config.t_per_anchor)
            test_loss.update(loss.item())
            
            spx_pools, super_feat = ungroup_batches(spx_pools, super_feat)
            
            for b in range(len(super_feat)):
            
                x = super_feat[b].clone().detach().cpu().numpy()
                y = spx_pools[b].clone().detach().cpu().numpy()
                idx = np.arange(1, y.shape[0]+1)
                
                try:                
                    x_train, x_test, y_train, y_test, idx_train, idx_test = \
                        train_test_split(x, y, idx, test_size=config.knn_test_size, stratify=y)                    
                    knn.fit(x_train,y_train)
                    #pred = knn.predict(x_test)            
                    knn_score.update(knn.score(x_test, y_test))
                except:
                    print('Knn skiped at current sample.')
                    continue
                
            t.set_postfix_str('loss: {:^7.3f}, knn_score: {:^7.3f}'.format(
                test_loss.avg, knn_score.avg))
            t.update()
    

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
    run(config)