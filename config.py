#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import argparse
import torch


class DefaultConfig(object):
    def __init__(self, args):
        
        self.experiment_name = args.exp
        self.save_model_path = os.path.join('./saved_models/', self.experiment_name)
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        #self.save_model_path = self.save_model_path + self.current_time
        self.epoch = 50
        
        # SPX_embedding
        self.feat_extractor_backbone = 'resnet18'
        self.sf_channels = 32 # super-feature's output channels
        self.f_size = 1
        self.k_size = 3
        
        self.loss = 'NTXentLoss'
        self.t_per_anchor = 5 #20 # triplets per anchor for computing loss
        self.classifier_optimizer = 'Adam'
        self.lr_scheduler = 'plateau'
        
        # Root folder of datasets
        #self.data_root = os.path.abspath('../databases')
        self.data_root = os.path.abspath(args.data_root)
        
        if torch.cuda.is_available():
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1
        
        if self.num_devices > 1:
            self.drop_last= True
        else:
            self.drop_last= False
        
        self.classifier_learning_rate = 1e-4 * self.num_devices 
        self.num_workers = 4 * self.num_devices
        self.test_times = 5
        self.save_model_times = 5
        self.early_test = False
        
        self.knn_neighbors = 5
        self.knn_test_size = 0.8


class ConfigForMSRA10K(DefaultConfig):
    def __init__(self, args):
        DefaultConfig.__init__(self, args)

        self.experiment_name = args.exp
        self.model = 'my_net'
        self.save_model_path = os.path.join('./saved_models/', self.experiment_name)
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        #self.save_model_path = self.save_model_path + self.current_time

        self.epoch = 200
        self.train_batch_size = 1# self.num_devices * 4
        self.test_batch_size = 1#self.num_devices * 4
        
        self.train_img_size=(256,256) #(384,384)
        self.val_img_size=(256,256) #(384,384)
        
        self.num_workers = self.num_devices #* 4
        self.test_times = 200000000000
        self.save_model_times = 100000000000
        
        # MSRA paths
        self.msra_root = os.path.join(self.data_root, 'MSRA', 'MSRA10K_Imgs_GT')
        self.msra_images = os.path.join(self.msra_root, 'Imgs')
        self.msra_masks = os.path.join(self.msra_root, 'saliency')
        
        self.msra_train_annotation = os.path.join(self.msra_root, 'train.txt')
        self.msra_val_annotation = os.path.join(self.msra_root, 'val.txt')
        
        # Superpixels
        self.pre_computed_spx = False
        self.spx_dir = None
        
        self.do_augmentation = True
        

class ConfigForDAVIS(DefaultConfig):
    def __init__(self, args):
        DefaultConfig.__init__(self, args)

        self.experiment_name = args.exp
        self.model = 'my_net'
        self.save_model_path = os.path.join('./saved_models/', self.experiment_name)
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        #self.save_model_path = self.save_model_path + self.current_time
        
        self.epoch = 200
        self.train_batch_size = 1 #self.num_devices * 4
        self.test_batch_size = 1 #self.num_devices * 4
        
        self.num_workers = self.num_devices #* 4
        self.test_times = 200000000000
        self.save_model_times = 100000000000

        self.max_img_side = 256 #480 # bigger image side ('None' preserves original size)
        
        # DAVIS paths
        self.davis_root = os.path.join(self.data_root, 'DAVIS2017')
        self.davis_images = os.path.join(self.davis_root, 'JPEGImages', '480p')
        self.davis_masks = os.path.join(self.davis_root, 'Annotations', '480p')
        
        self.davis_train_annotation = os.path.join(self.davis_root, 'ImageSets', '2017', 'train.txt')
        self.davis_val_annotation = os.path.join(self.davis_root, 'ImageSets', '2017', 'val.txt')
        
        # Superpixels
        self.pre_computed_spx = False
        self.spx_dir = None
        
        self.do_augmentation = True
        

################################################
def init(args):
    config = None
    if args.dataset.lower() == 'msra10k':
        config = ConfigForMSRA10K(args)
    elif args.dataset.lower() == 'davis':
        config = ConfigForDAVIS(args)

    # append args info into the config
    for k, v in vars(args).items():
        setattr(config, k, v)

    return config


def log_config(config):    
    if os.path.exists(config.save_model_path) is False:
        os.makedirs(config.save_model_path)    
    log_path = os.path.join(config.save_model_path, 
                            config.model + '_' + config.current_time + '.log')
    
    with open(log_path, 'w') as f:    
        for k, v in vars(config).items():
            f.write('{}: {}'.format(k,v))
            f.write('\n')
            #print('{}: {}'.format(k,v))
    

def set_config(jup_notebook=False, dataset='MSRA10k'):
    parser = argparse.ArgumentParser(description='My_Net Training') 
    #parser.add_argument("--dataset", type=str, default='MSRA_B')
    parser.add_argument("--dataset", type=str, default='MSRA10K')
    #parser.add_argument("--dataset", type=str, default='DAVIS')
    parser.add_argument("--exp", type=str, default='')
    parser.add_argument("--resume_model_path", type=str, default='')
    parser.add_argument('--gpu_id', type=str, default='', help='gpu id')
    parser.add_argument("--data_root", type=str, default='../databases', help='the dir of dataset')
    
    if not jup_notebook:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=[])
        args.dataset = dataset

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # init
    config = init(args)
    log_config(config)
    return config