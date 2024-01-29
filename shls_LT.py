#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from config_LT import set_config
import Data.get_data_LT as _Data
import Models.get_model_LT as _Models
from metrics_LT import get_metrics
from Data.mask_io import save_mask
from utils_LT import launch_cuda_ddp, get_spx_pools, load_checkpoint, save_model,\
        freeze_batchnorm, AverageMeter, format_time, show_intro, set_seed, context, to_onehot,\
            get_confident_pools, tensor_train_test_split


import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

def prepare_inputs(x, y, num_cls):    
    x_list, y_list = [], []    
    for k in range(1,num_cls+1):        
        x_list.append(x[y==k][:2])
        y_list.append(y[y==k][:2])    
    x_train, _, y_train, _, _, _ = tensor_train_test_split(x, y, test_size=0.4)
    x_list.append(x_train)
    y_list.append(y_train)        
    return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)
    

def train(config, model, data_loader, loss_function, optimizer, memory, epoch=0, it=0, writer=None, rank=None, world_size=None):
   
    model.train()
    freeze_batchnorm(model, config.freeze_modules)
    
    loss_ML_meter = AverageMeter()
    loss_CE_meter = AverageMeter()
    pre_J = AverageMeter()
    pre_F = AverageMeter()
    fin_J = AverageMeter()
    fin_F = AverageMeter()
    t = iter(data_loader)
    seq_len = config.seq_len
    
    
    if rank == 0:
        t = tqdm(t)
        t.set_description("Train [ep={}/{}]".format(epoch+1,config.epoch))
    
    for i, (img_seq, spx_seq, label_seq, num_cls, info) in enumerate(t):        
        # seq: torch.Size([b, seq_len, c, h, w])
        
        it += 1
        
        assert img_seq.shape[0] == 1, 'Minibach > 1 is not supported. Try sequence lenght.'              

        # input image, superpixels and pseudo-labels
        img_seq = img_seq.cuda(rank)
        spx_seq = spx_seq.cuda(rank)
        label_seq = label_seq.cuda(rank)
        
        num_cls = num_cls[0].item()  
        loss_ML = 0.0
        loss_CE = 0.0
        
        ##### 1st Frame
        
        # pools of superpixels per class
        spx_pools, spx_seq[:,0] = get_spx_pools(spx_seq[:,0], label_seq[:,0], merge=True)
        #spx_pools = get_spx_pools(spx_seq[:,0], label_seq[:,0], merge=False)
        
        # make super features from superpixels and embeddings
        with context('superfeat' in config.no_grad, torch.no_grad()):
            superft = model(img_seq[:,0], spx_seq[:,0].float(), n=0, mod='superfeat')
        
        # compute loss with metric learning
        loss_ML += loss_function.metric_learning(spx_pools, superft)
        
        # fit memory clusterer
        memory.fit(superft[0].detach().clone(), spx_pools[0].clone(), num_cls)        
        #memory.fit(*prepare_inputs(superft[0].detach().clone(), spx_pools[0].clone(), num_cls), num_cls)        
        
        if 'train' in config.save_mask:
            save_mask(config, label_seq[0,0,0], info['name'][0], 0, frame=img_seq[0,0], prefix='preseg')
        
        lastseg = label_seq[:,0].clone()
        lastpred = to_onehot(lastseg, num_cls)
        
        _, attmaps_0, bbox_0 = memory.predict(superft[0].detach().clone(), 
                                                             spx_seq[:,0].clone().int(),
                                                             lastseg, lastpred,
                                                             gt=label_seq[:,0])  
        
        
        ##### next frames        
        for n in range(1, seq_len):
            
            spx_pools = get_spx_pools(spx_seq[:,n], label_seq[:,n], merge=False)
            
            with context('superfeat' in config.no_grad, torch.no_grad()):
                superft = model(img_seq[:,n], spx_seq[:,n].float(), mod='superfeat')
                
            loss_ML += loss_function.metric_learning(spx_pools, superft)
            
            # get an attention map and pre-segmentation mask from memory            
            preseg, attmaps, bbox = memory.predict(superft[0].detach().clone(), 
                                                             spx_seq[:,n].clone().int(),
                                                             lastseg, lastpred,
                                                             gt=label_seq[:,n])  
            
            # predict the final segmentation
            with context('segment' in config.no_grad, torch.no_grad()):                
                pred = model(attmaps_0, attmaps, 
                             bbox_0, bbox, 
                             lastpred, num_cls, mod='segment')
                
                
            attmaps_0, bbox_0 = attmaps.copy(), bbox.copy()
            
            # compute cross entropy loss pixelwisely
            loss_CE += loss_function.cross_entropy(pred, (label_seq[:,n,0]-1).long())            
            
            
            lastpred = pred.detach().clone()
            seg = (torch.argmax(lastpred, dim=1).int() + 1).unsqueeze(dim=1)
            lastseg = seg.clone()
            #lastseg = label_seq[:,n].clone()
            
            
            # update memory clusterer
            if n < seq_len-1:
                #seg_pools = get_spx_pools(spx_seq[:,n], seg, merge=False)                        
                #memory.update(superft[0].detach().clone(), seg_pools[0])
                #memory.update(superft[0].detach().clone(), spx_pools[0])
                mem_pools = get_confident_pools(spx_seq[:,n], preseg, lastpred, num_cls)
                if mem_pools:
                    seg_pools = get_spx_pools(spx_seq[:,n], preseg, merge=False)
                    memory.update(superft[0].detach().clone()[mem_pools[0]==1], seg_pools[0][mem_pools[0]==1])
                
            
            # compute Jaccard and F-score metrics
            p_J, p_F = get_metrics(preseg, label_seq[:,n], num_cls=num_cls)
            f_J, f_F = get_metrics(seg, label_seq[:,n], num_cls=num_cls)            
            pre_J.update(p_J)
            pre_F.update(p_F)
            fin_J.update(f_J)
            fin_F.update(f_F)
            
            if 'train' in config.save_mask:
                save_mask(config, preseg[0], info['name'][0], n, frame=img_seq[0,n], prefix='preseg')
            
        # compute loss
        loss_ML = loss_ML/seq_len
        loss_CE = loss_CE/(seq_len-1)        
        loss_ML_meter.update(loss_ML.item())
        loss_CE_meter.update(loss_CE.item())
        loss = loss_ML + loss_CE

        # backward and update
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()               
            optimizer.step()        
        
        # gather distributed metrics 
        dist_itens = [pre_J.avg, pre_F.avg, fin_J.avg, fin_F.avg, loss_ML_meter.avg, loss_CE_meter.avg]
        all_itens = [None for _ in range(world_size)]
        dist.all_gather_object(all_itens, dist_itens)
        all_p_J = np.mean([x[0] for x in all_itens])
        all_p_F = np.mean([x[1] for x in all_itens])
        all_f_J = np.mean([x[2] for x in all_itens])
        all_f_F = np.mean([x[3] for x in all_itens])
        all_loss_ML = np.mean([x[4] for x in all_itens])
        all_loss_CE = np.mean([x[5] for x in all_itens])
        all_p_JF = (all_p_J+all_p_F)/2
        all_f_JF = (all_f_J+all_f_F)/2
        
        # write the summary
        if rank == 0:               
            if writer is not None:
                writer.add_scalar('Test: pre-J&F', all_p_JF, it)
                writer.add_scalar('Test: final-J&F', all_f_JF, it)
                writer.add_scalar('Test: ML Loss', all_loss_ML, it)
                writer.add_scalar('Test: CE Loss', all_loss_CE, it)
            
            t.set_postfix_str('Pre:ML-loss={:^7.3f},_J&F={:^7.2f}_|_Fin:CE-loss={:^7.3f},_J&F={:^7.2f}'.format(
                                                  all_loss_ML, all_p_JF, all_loss_CE, all_f_JF).replace(" ", "").replace("_", " "))
            t.update()
    
    return it, all_loss_ML+all_loss_CE
        
        
def test(config, model, data_loader, loss_function, memory, epoch=0, it=0, writer=None, rank=None, world_size=None):
    
    test_time = time.time()
    set_seed()
    model.eval()
    with torch.no_grad():        
        loss_ML = AverageMeter()
        loss_CE = AverageMeter()
        pre_J = AverageMeter()
        pre_F = AverageMeter()
        fin_J = AverageMeter()
        fin_F = AverageMeter()
        spx_n = AverageMeter()
        #video_res = {}
        
        
        t = iter(data_loader)
        
        
        with model.join() and model.no_sync():
            for i, (img_seq, spx_seq, label_seq, num_cls, info) in enumerate(t):
            # seq: torch.Size([b, seq_len, c, h, w])
                
                it += 1
                
                v_J = AverageMeter()
                v_F = AverageMeter()
            
                assert img_seq.shape[0] == 1, 'Minibach > 1 is not supported for video.'
                
                num_cls = num_cls[0].item()
        
                # input image, superpixels and pseudo-labels
                img_seq = img_seq.cuda(rank)
                spx_seq = spx_seq.cuda(rank)
                label_seq = label_seq.cuda(rank)
                
                seq_len = info['num_frames'][0].item()
                #seq_len = int(seq_len/8)
                
                ##### 1st Frame
                
                spx_n.update(spx_seq[:,0].max().item())
                # pools of superpixels per class
                spx_pools, spx_seq[:,0] = get_spx_pools(spx_seq[:,0], label_seq[:,0], merge=True)
                #spx_pools = get_spx_pools(spx_seq[:,0], label_seq[:,0], merge=False)
                
                # make super features from superpixels and embeddings
                superft = model(img_seq[:,0], spx_seq[:,0].float(), n=0, mod='superfeat')
                
                # fit memory clusterer                
                memory.fit(superft[0].clone(), spx_pools[0].clone(), num_cls)
                #memory.fit(superft[0].clone(), spx_pools[0].clone(), spx_seq[0,0,0])
                
                # compute loss with metric learning
                loss_ML.update(loss_function.metric_learning(spx_pools, superft).item())
                
                # leverage the given 1st annotation
                lastseg =  label_seq[:,0].clone()
                lastpred = to_onehot(lastseg, num_cls)
                
                _, attmaps_0, bbox_0 = memory.predict(superft[0].clone(), 
                                                             spx_seq[:,0].clone().int(),
                                                             lastseg, lastpred,
                                                             gt=label_seq[:,0])
                                                             #nm=info['name'][0], nn=0) 
                
                if 'test' in config.save_mask:
                    #save_mask(config, label_seq[:,0], info['name'][0], 0, frame=img_seq[:,0], prefix='preseg')
                    #save_mask(config, label_seq[:,0], info['name'][0], 0, frame=img_seq[:,0], prefix='seg')
                    save_mask(config, label_seq[:,0], info['name'][0], 0, frame=None, prefix='preseg')
                    save_mask(config, label_seq[:,0], info['name'][0], 0, frame=None, prefix='seg')
                
                ##### next frames
                N = list(range(1, seq_len))
                N = tqdm(N, position=rank, leave=False)
                N.set_description("Test[ep={}][GPU:{}]V={}/{}".format(
                    epoch, rank, i+1, len(data_loader)).replace(" ", ""))                    
                
                for n in N:            
                    spx_n.update(spx_seq[:,n].max().item())
                    spx_pools = get_spx_pools(spx_seq[:,n], label_seq[:,n], merge=False)
                    
                    superft = model(img_seq[:,n], spx_seq[:,n].float(), mod='superfeat')
                    
                    loss_ML.update(loss_function.metric_learning(spx_pools, superft).item())
                    
                    preseg, attmaps, bbox = memory.predict(superft[0].clone(), 
                                                             spx_seq[:,n].clone().int(),
                                                             lastseg, lastpred,)
                                                             #gt=label_seq[:,n],)
                                                             #nm=info['name'][0], nn=n)
                                                             #gt=img_seq[:,n-1:n+1])          
                    
                    
                    # predict the final segmentation
                    #pred = model(attmaps, bbox, lastpred, img_seq[:,n].clone(), num_cls, mod='segment')                    
                    pred = model(attmaps_0, attmaps, 
                             bbox_0, bbox, 
                             lastpred, num_cls, mod='segment')
                
                
                    attmaps_0, bbox_0 = attmaps.copy(), bbox.copy()
                    
                    
                    # compute cross entropy loss pixelwisely
                    loss_CE.update(loss_function.cross_entropy(pred, (label_seq[:,n,0]-1).long()).item())
                   
                    seg = (torch.argmax(pred, dim=1).int() + 1).unsqueeze(dim=1)
                    lastseg = seg.clone()
                    #lastseg = preseg.clone()
                    lastpred = pred.clone()
                    
                    # update memory clusterer
                    if  n < seq_len-1:
                        mem_pools = get_confident_pools(spx_seq[:,n], seg, pred, num_cls)
                        if mem_pools:
                            seg_pools = get_spx_pools(spx_seq[:,n], preseg, merge=False)
                            memory.update(superft[0][mem_pools[0]==1], seg_pools[0][mem_pools[0]==1])
                        # seg_pools = get_spx_pools(spx_seq[:,n], preseg, merge=False)
                        # memory.update(superft[0], seg_pools[0])                        
                        #spx_pools = get_spx_pools(spx_seq[:,n], label_seq[:,n], merge=False)
                        #memory.update(superft[0], spx_pools[0])
                    
                    
                    # compute Jaccard and F-score metrics
                    p_J, p_F = get_metrics(preseg, label_seq[:,n], num_cls=num_cls)
                    f_J, f_F = get_metrics(seg, label_seq[:,n], num_cls=num_cls)
                    pre_J.update(p_J)
                    pre_F.update(p_F)
                    fin_J.update(f_J)
                    fin_F.update(f_F)
                    
                    v_J.update(f_J)
                    v_F.update(f_F)
                    
                    if True: # rank == 0:                    
                        N.set_postfix_str('Pre:J={:^7.2f},F={:^7.2f},J&F={:^7.2f}_|_Fin:J={:^7.2f},F={:^7.2f},J&F={:^7.2f}'.format(
                                                pre_J.avg, pre_F.avg, (pre_J.avg+pre_F.avg)/2,
                                                fin_J.avg, fin_F.avg, (fin_J.avg+fin_F.avg)/2).replace(" ", "").replace("_", " "))                        
                        N.update()
                        
                    if 'test' in config.save_mask:
                        #save_mask(config, preseg, info['name'][0], n, frame=img_seq[:,n], prefix='preseg')
                        #save_mask(config, seg, info['name'][0], n, frame=img_seq[:,n], prefix='seg')
                        save_mask(config, preseg, info['name'][0], n, frame=None, prefix='preseg')
                        save_mask(config, seg, info['name'][0], n, frame=None, prefix='seg')
                
                #video_res[info['name'][0]] = (v_J.avg+v_F.avg)/2
                
                N.close()
                
                
        # gather distributed metrics 
        dist_itens = [pre_J.avg, pre_F.avg, fin_J.avg, fin_F.avg, loss_ML.avg, loss_CE.avg, spx_n.avg]
        all_itens = [None for _ in range(world_size)]
        dist.all_gather_object(all_itens, dist_itens)
        all_p_J = np.mean([x[0] for x in all_itens])
        all_p_F = np.mean([x[1] for x in all_itens])
        all_f_J = np.mean([x[2] for x in all_itens])
        all_f_F = np.mean([x[3] for x in all_itens])
        all_loss_ML = np.mean([x[4] for x in all_itens])
        all_loss_CE = np.mean([x[5] for x in all_itens])
        all_p_JF = (all_p_J+all_p_F)/2
        all_f_JF = (all_f_J+all_f_F)/2
        all_spx_n = np.mean([x[6] for x in all_itens])
        
        
        out_str = ''
        if rank == 0:               
            if writer is not None:
                writer.add_scalar('Test: pre-J&F', all_p_JF, it)
                writer.add_scalar('Test: final-J&F', all_f_JF, it)
                writer.add_scalar('Test: ML Loss', all_loss_ML, it)
                writer.add_scalar('Test: CE Loss', all_loss_CE, it)
            
            e_time = format_time(time.time()-test_time)            
            out_str = 'Test[epoch={}][{}]:_'.format(epoch, e_time).replace(" ", "")
            out_str += '[Pre:ML-loss={:^7.3f},J={:^7.3f},F={:^7.3f},J&F={:^7.3f}'.format(all_loss_ML, all_p_J, all_p_F, all_p_JF).replace(" ", "")
            out_str += '_|_Fin:CE-loss={:^7.3f},J={:^7.3f},F={:^7.3f},J&F={:^7.3f}]'.format(all_loss_CE, all_f_J, all_f_F, all_f_JF)
            out_str = out_str.replace(" ", "").replace("_", " ")
            
            print('\nFinal result: \n-----------------------------')
            print(out_str+'                                 ')
            print('\nSPX={:^7.2f}'.format(all_spx_n))
            print('-----------------------------\n')
        
        #print('rank: ', rank, '\n', video_res)
        
        return it, all_f_JF, out_str
        #return it, all_p_JF, out_str


def run(rank, config):
    
    world_size = config.num_devices
    setup(rank, world_size)
        
    # data 
    train_loader, test_loader   = _Data.get_data_ddp(config, rank, world_size)
    # model & loss
    model, loss_function        = _Models.get_model_loss(config, rank)
    # optimizer & scheduler
    optimizer, lr_scheduler     = _Models.get_opti_scheduler(config, model)
    # memory clusterer
    #train_mem, test_mem = _Models.get_memory(config), _Models.get_memory(config.config_test)
    
    start_epoch = 0
    train_it = 0    
    test_it = 0    
    JF = 0.0 # mean Jaccard + F-score
    top_JF = 0.0
    train_loss = 0.0
    writer = None
    
    # tensorboard    
    if rank == 0 and config.tensorboard:
        writer = SummaryWriter(config.save_model_path+'/t_board')
    
    # resume from checkpoint
    if config.resume_model_path:
        #model, optimizer, start_epoch, top_JF, train_it, test_it = load_checkpoint(config, model, optimizer, rank=rank, train=True)
        model, start_epoch, top_JF, train_it, test_it = load_checkpoint(config, model, rank=rank, train=True)
    
    # send model and loss to gpu(s)
    model, optimizer = launch_cuda_ddp(model, rank, optimizer, broadcast_buffers=False)
    loss_function = loss_function.cuda(rank)           
    
    # train-test loop
    for epoch in range(start_epoch, config.epoch):
        
        # set up ddp data samplers
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        train_mem, test_mem = _Models.get_memory(config), _Models.get_memory(config.config_test)
        
        # test before start training
        if config.early_test and epoch == start_epoch:
            test_it, _, _ = test(config, model, test_loader, loss_function, 
                              test_mem, epoch, test_it, writer, rank, world_size)
            break
        
        # train
        train_it, train_loss = train(config, model, train_loader, loss_function,
                                      optimizer, train_mem, epoch, train_it, writer, rank, world_size)
        #JF = train_loss
        # test
        test_it, JF, log = test(config, model, test_loader, loss_function,
                            test_mem, epoch+1, test_it, writer, rank, world_size) 
        
        
        # update learning rate
        if lr_scheduler is not None: lr_scheduler.step(JF)
        
        # save checkpoint
        if rank == 0:
            if JF > top_JF:
                top_JF = JF
                save_model(config, model, '_best.pth', epoch, optimizer, JF, top_JF, train_it, test_it, train_loss)
                optimizer, lr_scheduler     = _Models.get_opti_scheduler(config, model)
            e_time = format_time(time.time()-start_time)
            save_model(config, model, '_last.pth', epoch, optimizer, JF, top_JF, train_it, test_it, train_loss, e_time, log)
            print("Finished epoch: [{}/{}]".format(epoch+1,config.epoch)+' - Elapsed time: '+ e_time)
    
    print('Rank {} completed!'.format(rank))
    
   

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)    
    
    
if __name__ == '__main__':
    config = set_config()    
    show_intro(config)
    
    try:
        mp.spawn(run, nprocs=config.num_devices, args=(config,))
    except:
        raise RuntimeError('Unable to start Distributed Dataparallel (DDP) processes.')


# if __name__ == '__main__':
    
    
#     k = [2, 4, 8, 16, 32, 64]
    
#     for i in range(len(k)):
#         print('\n >>>>>>> [{}/{}] nkc = {}'.format(i+1, len(k), k[i]))
#         config = set_config()
#         config.config_test.nkc = k[i]
#         config.nkc = k[i]
#         #config.config_test.slic_num = k[i]
        
#         show_intro(config)
        
#         try:
#             mp.spawn(run, nprocs=config.num_devices, args=(config,))
#         except:
#             raise RuntimeError('Unable to start Distributed Dataparallel (DDP) processes.')

