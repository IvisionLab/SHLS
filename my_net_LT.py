#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from config_LT import set_config
import Data.get_data_LT as _Data
import Models.get_model_LT as _Models
from utils_LT import launch_cuda_ddp, get_spx_pools, load_checkpoint, save_model\
                    ,AverageMeter, spx_iou, FaissKNeighbors, show_intro\
                    , format_time\


start_time = time.time()


def train(config, model, data_loader, loss_function, optimizer, epoch=0, it=0, writer=None, rank=None, world_size=None):
    
    # set train mode
    model.train()
    
    loss_meter = AverageMeter() 
    iou = AverageMeter()
    t = iter(data_loader)
    seq_len = config.seq_len
    
    if rank == 0:
        t = tqdm(t)
        t.set_description("Train [epoch={}/{}]".format(epoch+1,config.epoch))
    
    
    for i, (img_seq, spx_seq, label_seq, _, _) in enumerate(t):
        # seq: torch.Size([b, seq_len, c, w, h])
        it += 1
        
        ######## knn train
        knn = []
        batch_size = spx_seq.shape[0]

        # input image, generated superpixels and object pseudo-labels
        img_seq = img_seq.cuda(rank)
        spx_seq = spx_seq.cuda(rank)
        label_seq = label_seq.cuda(rank)
        
        
        for b in range(batch_size):            
            knn.append(FaissKNeighbors(k=config.knn_neighbors))
        
        loss = 0.0
        for n in range(seq_len):            
            # pools of superpixels by object
            spx_pools, _ = get_spx_pools(spx_seq[:,n], label_seq[:,n], merge=(n==0))
            
            # make super features from superpixels and embeddings
            super_feat = model(img_seq[:,n], spx_seq[:,n].float())
            
            
            # compute loss with metric learning
            #loss = loss + loss_function(spx_pools.copy(), super_feat.copy())
            loss = loss + loss_function(spx_pools, super_feat)
            
            
            ######## knn train
            for b in range(batch_size):
                x = super_feat[b].detach().clone()#.cpu().numpy()
                y = spx_pools[b].detach().clone()#.cpu().numpy()
                idx = torch.arange(1, y.shape[0]+1)
                
                if n == 0:
                    knn[b].fit(x, y)
                else:
                    pred = knn[b].predict(x)
                    _iou = spx_iou(label_seq[b,n,0].clone(), spx_seq[b,n,0].clone(), pred.cuda(rank), [], [], idx)
                    if _iou is not None: iou.update(_iou)
        
        loss = loss/seq_len

        # backward and update
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

        # gather distributed losses at gpu rank == 0
        loss_outputs = [None for _ in range(world_size)]
        dist.all_gather_object(loss_outputs, loss.item())
        if rank == 0:
            for l in loss_outputs:
                loss_meter.update(l)
        
        # write the summary
        if rank == 0:
            if writer is not None: writer.add_scalar('Train: loss_avg', loss_meter.avg, it)                
            t.set_postfix_str('loss:{:^7.3f} IoU:{:^7.3f}'.format(loss_meter.avg, iou.avg))
            t.update()
        
    return it, loss_meter.avg
   
        
        
def test(config, model, data_loader, loss_function, epoch=0, it=0, writer=None, rank=None, world_size=None):
    model.eval()
    
    with torch.no_grad():                
        loss_meter = AverageMeter() 
        knn_score = AverageMeter()
        iou = AverageMeter()
        knn = FaissKNeighbors(k=config.knn_neighbors)
        skipped_samples = 0
        
        t = iter(data_loader)
        if rank == 0:
            t = tqdm(t)        
            t.set_description("Test [epoch={}]".format(epoch))
        
        assert '1st_frame' in config.config_test.to_return
        for i, (img, spx, label, num_cls, info, img_1st, spx_1st, label_1st) in enumerate(t):
            it += 1
            
            img, spx, label = img.float().cuda(rank), spx.cuda(rank), label.cuda(rank)
            img_1st, spx_1st, label_1st = img_1st.float().cuda(rank), spx_1st.cuda(rank), label_1st.cuda(rank)
            
            spx_pools, _ = get_spx_pools(spx, label)
            super_feat = model(img, spx.float())            
            
            # 1st frame stuff
            spx_pools_1st, _ = get_spx_pools(spx_1st, label_1st, merge=True)
            super_feat_1st = model(img_1st, spx_1st.float())            
            
            loss = loss_function(spx_pools.copy(), super_feat.copy())
            loss_meter.update(loss.item())
            
            for b in range(len(super_feat)):
                if info['frame_idx'][b].item() == 0:
                    continue
                # invalid sample
                if num_cls[b] < 2:
                    skipped_samples += 1
                    continue
                x = super_feat[b].detach().clone()#.cpu()#.numpy()
                y = spx_pools[b].detach().clone()#.cpu()#.numpy()
                idx = torch.arange(1, y.shape[0]+1)
                
                # 1st frame stuff
                x_1st = super_feat_1st[b].detach().clone()#.cpu().numpy()
                y_1st = spx_pools_1st[b].detach().clone()#.cpu().numpy()
                #idx_1st = torch.arange(1, y_1st.shape[0]+1)
                
                knn.fit(x_1st, y_1st)
                
                pred, score = knn.predict(x, y)
                knn_score.update(score.item())
                _iou = spx_iou(label[b,0].clone(), spx[b,0].clone(), pred.cuda(rank), [], [], idx)
                if _iou is not None: iou.update(_iou)
            
            loss_outputs = [None for _ in range(world_size)]
            knn_outputs = [None for _ in range(world_size)]
            iou_outputs = [None for _ in range(world_size)]
            skipped_samples_outputs = [None for _ in range(world_size)]
            i_outputs = [None for _ in range(world_size)]
            
            dist.all_gather_object(loss_outputs, loss_meter.avg)
            dist.all_gather_object(knn_outputs, knn_score.avg)
            dist.all_gather_object(iou_outputs, iou.avg)
            dist.all_gather_object(skipped_samples_outputs, skipped_samples)
            dist.all_gather_object(i_outputs, i+1)
            
            if rank == 0:
                all_test_loss = np.mean(loss_outputs)
                all_knn_score = np.mean(knn_outputs)
                all_iou = np.mean(iou_outputs)
                all_skipped_samples = np.sum(skipped_samples_outputs)
                all_i = np.sum(i_outputs)                
            
                if writer is not None:
                    writer.add_scalar('Test: loss_avg', all_test_loss, it)
                    writer.add_scalar('Test: knn_score', all_knn_score, it)
                    writer.add_scalar('Test: IoU', all_iou, it)
                
                t.set_postfix_str('loss:{:^7.3f}knn:{:^7.3f}IoU:{:^7.3f}'.format(
                    all_test_loss, all_knn_score, all_iou))
                t.update()
            else:
                all_iou = np.mean(iou_outputs)
    
    if rank == 0 and all_skipped_samples > 0:
        t.write('Knn: {}/{} samples skipped at test.'.format(all_skipped_samples, all_i))
    
    return it, all_iou


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
    
    start_epoch = 0
    train_it = 0    
    test_it = 0
    top_iou = 0.0
    writer = None
    
    # tensorboard    
    if rank == 0 and config.tensorboard:
        writer = SummaryWriter(config.save_model_path+'/t_board')
    
    # resume from checkpoint
    if config.resume_model_path:
        model, optimizer, start_epoch, top_iou, train_it, test_it = load_checkpoint(config, model, optimizer, rank, train=True)
    
    # send model and loss to gpu(s)
    model, optimizer = launch_cuda_ddp(model, rank, optimizer, broadcast_buffers=False)
    loss_function = loss_function.cuda(rank)           
    
    # train-test loop
    for epoch in range(start_epoch, config.epoch):
        
        # set up ddp data samplers
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        
        # test before start training
        if config.early_test and epoch == start_epoch:
            test_it, _ = test(config, model, test_loader, loss_function, 
                              epoch, test_it, writer, rank, world_size)
        
        # train
        #torch.cuda.empty_cache()        
        train_it, train_loss = train(config, model, train_loader, loss_function,
                                     optimizer, epoch, train_it, writer, rank, world_size)
        
         # test 
        #torch.cuda.empty_cache()
        test_it, iou = test(config, model, test_loader, loss_function,
                            epoch+1, test_it, writer, rank, world_size)
        
        
        # update learning rate
        if lr_scheduler is not None: lr_scheduler.step(iou)
        
        # save checkpoint
        if rank == 0:
            if iou > top_iou:
                top_iou = iou
                save_model(config, model, '_best.pth', epoch, optimizer, iou, top_iou, train_it, test_it, train_loss)
            e_time = format_time(time.time()-start_time)
            save_model(config, model, '_last.pth', epoch, optimizer, iou, top_iou, train_it, test_it, train_loss, e_time)
            print("Finished epoch: [{}/{}]".format(epoch+1,config.epoch)+' - Elapsed time: '+ e_time)
   

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


