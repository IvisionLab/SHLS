#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import argparse
import torch


class DefaultConfig(object):
    def __init__(self):
        self.root = './'
        self.feat_extractor_backbone = 'resnet18'
        self.save_model_path = './saved_models/'
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        self.save_model_path = self.save_model_path + self.current_time
        self.epoch = 50
        
        self.loss = 'NTXentLoss'
        self.t_per_anchor = 10 # triplets per anchor for computing loss
        self.classifier_optimizer = 'Adam'
        self.classifier_learning_rate = 1e-4
        
        # Root folder of datasets
        self.data_root = os.path.abspath('../databases')
        
        if torch.cuda.is_available():
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1
        
        if self.num_devices > 1:
            self.drop_last= True
        else:
            self.drop_last= False
        
        self.num_workers = 2 * self.num_devices
        self.test_times = 5
        self.save_model_times = 5
        
        self.knn_neighbors = 5
        self.knn_test_size = 0.7


class ConfigForMSRA10K(DefaultConfig):
    def __init__(self):
        DefaultConfig.__init__(self)

        self.model = 'my_net'
        self.save_model_path = './saved_models/MSRA10K/'
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        self.save_model_path = self.save_model_path + self.current_time

        self.epoch = 200
        self.train_batch_size = 2
        self.test_batch_size = 2
        self.train_batch_size *= self.num_devices
        self.test_batch_size *= self.num_devices
        
        self.train_img_size=(256,256) #(384,384)
        self.val_img_size=(256,256) #(384,384)
        
        # MSRA paths
        self.msra_root = os.path.join(self.data_root, 'MSRA', 'MSRA10K_Imgs_GT')
        self.msra_masks = os.path.join(self.msra_root, 'Imgs')
        self.msra_images = os.path.join(self.msra_root, 'Imgs')
        
        # Superpixels
        self.pre_computed_spx = False
        self.spx_dir = None

class ConfigForMSRA_B(DefaultConfig):
    def __init__(self):
        DefaultConfig.__init__(self)

        self.model = 'my_net'
        self.save_model_path = './saved_models/MSRA_B/'
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        self.save_model_path = self.save_model_path + self.current_time

        self.epoch = 200
        self.train_batch_size = 2
        self.test_batch_size = 2
        self.train_batch_size *= self.num_devices
        self.test_batch_size *= self.num_devices
        
        self.train_img_size=(256,256) #(384,384)
        self.val_img_size=(256,256) #(384,384)
        
        # MSRA paths
        self.msra_root = os.path.join(self.data_root, 'MSRA', 'MSRA_B')
        #self.msra_masks = os.path.join(self.msra_root, 'saliency')
        self.msra_images = os.path.join(self.msra_root, 'images')
        
        self.msra_train_annotation = os.path.join(self.msra_root, 'train_4.5k.txt')
        #self.msra_val_annotation = os.path.join(self.msra_root, 'val_0.5k.txt')
        self.msra_val_annotation = os.path.join(self.msra_root, 'val_20.txt')
        
        self.saliency_maps = ['03_mc', '04_hs', '05_dsr', '06_rbd', 'jeff_all']
        self.to_use_saliency_maps = [4]
        
        self.msra_masks = []
        for m in self.to_use_saliency_maps:
            self.msra_masks.append(os.path.join(self.msra_root, 'saliency', self.saliency_maps[m]))            
        
        # Superpixels
        self.pre_computed_spx = False
        self.spx_dir = None

class ConfigForMSRA_B_sdumont(DefaultConfig):
    def __init__(self):
        DefaultConfig.__init__(self)

        self.model = 'my_net_part2'
        self.save_model_path = './saved_models/MSRA_B/'
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        #self.save_model_path = self.save_model_path + self.current_time

        self.epoch = 200
        self.train_batch_size = 4
        self.test_batch_size = 4 
        self.train_batch_size *= self.num_devices
        self.test_batch_size *= self.num_devices
        
        self.train_img_size=(256,256) #(384,384)
        self.val_img_size=(256,256) #(384,384)
        
        self.num_workers = 16
        self.test_times = 2000
        self.save_model_times = 1000000000
        
        # Root folder of datasets
        self.data_root = os.path.abspath('../my_datasets')
        
        # MSRA paths
        self.msra_root = os.path.join(self.data_root, 'MSRA', 'MSRA_B')
        #self.msra_masks = os.path.join(self.msra_root, 'saliency')
        self.msra_images = os.path.join(self.msra_root, 'images')
        
        self.msra_train_annotation = os.path.join(self.msra_root, 'train_4.5k.txt')
        self.msra_val_annotation = os.path.join(self.msra_root, 'val_0.5k.txt')
        #self.msra_val_annotation = os.path.join(self.msra_root, 'val_20.txt')
        
        self.saliency_maps = ['03_mc', '04_hs', '05_dsr', '06_rbd', 'jeff_all']
        self.to_use_saliency_maps = [4]
        
        self.msra_masks = []
        for m in self.to_use_saliency_maps:
            self.msra_masks.append(os.path.join(self.msra_root, 'saliency', self.saliency_maps[m]))            
        
        # Superpixels
        self.pre_computed_spx = True
        self.spx_dir = os.path.join(self.msra_root, 'superpixels', 'isec_masks')
        
class ConfigForMSRA_B_pinha(DefaultConfig):
    def __init__(self):
        DefaultConfig.__init__(self)

        self.model = 'my_net_pinha'
        self.save_model_path = './saved_models/MSRA_B/'
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        #self.save_model_path = self.save_model_path + self.current_time

        self.epoch = 200
        self.train_batch_size = 4
        self.test_batch_size = 4 
        self.train_batch_size *= self.num_devices
        self.test_batch_size *= self.num_devices
        
        self.train_img_size=(256,256) #(384,384)
        self.val_img_size=(256,256) #(384,384)
        
        self.classifier_learning_rate *= self.train_batch_size/32 # 32 is the batch_size=4 * 8 gpus in sdumont
        
        self.num_workers = 16
        self.test_times = 2000
        self.save_model_times = 1000000000
        
        # Root folder of datasets
        self.data_root = os.path.abspath('../my_datasets')
        
        # MSRA paths
        self.msra_root = os.path.join(self.data_root, 'MSRA', 'MSRA_B')
        #self.msra_masks = os.path.join(self.msra_root, 'saliency')
        self.msra_images = os.path.join(self.msra_root, 'images')
        
        self.msra_train_annotation = os.path.join(self.msra_root, 'train_4.5k.txt')
        self.msra_val_annotation = os.path.join(self.msra_root, 'val_0.5k.txt')
        #self.msra_val_annotation = os.path.join(self.msra_root, 'val_20.txt')
        
        self.saliency_maps = ['03_mc', '04_hs', '05_dsr', '06_rbd', 'jeff_all']
        self.to_use_saliency_maps = [4]
        
        self.msra_masks = []
        for m in self.to_use_saliency_maps:
            self.msra_masks.append(os.path.join(self.msra_root, 'saliency', self.saliency_maps[m]))            
        
        # Superpixels
        self.pre_computed_spx = True
        self.spx_dir = os.path.join(self.msra_root, 'superpixels', 'isec_masks')


def init(args):
    config = None
    if args.dataset.lower() == 'msra10k':
        config = ConfigForMSRA10K()
    elif args.dataset.lower() == 'msra_b':
        config = ConfigForMSRA_B()
    elif args.dataset.lower() == 'msra_b_sdumont':
        config = ConfigForMSRA_B_sdumont()
    elif args.dataset.lower() == 'msra_b_pinha':
        config = ConfigForMSRA_B_pinha()

    # append args info into the config
    for k, v in vars(args).items():
        setattr(config, k, v)

    return config

def set_config():
    parser = argparse.ArgumentParser(description='My_Net Training') 
    #parser.add_argument("--dataset", type=str, default='MSRA10K')
    parser.add_argument("--dataset", type=str, default='MSRA_B')
    parser.add_argument("--resume_model_path", type=str, default='')
    parser.add_argument('--gpu_id', type=str, default='', help='gpu id')
    parser.add_argument("--data_root", type=str, default='MSRA',help='the dir of dataset')
    args = parser.parse_args()
    #args = parser.parse_args(args=[])

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    

    # init
    config = init(args)
    return config