#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import argparse
import torch


class DefaultConfig(object):
    def __init__(self, args):
        
        # Device configurations
        self.num_devices = 1
        if torch.cuda.is_available(): self.num_devices = torch.cuda.device_count()
        
        # Model parameters
        self.feat_extractor_backbone = 'resnet18'
        self.sf_channels = 32 # super-feature's output channels
        self.freeze_modules = ['embedding', 'sf', 'segmenter'] # opstions: 'embedding', 'sf', 'mem', 'segmenter'
        self.no_grad = [] # options: 'superfeat', 'segment'
        self.bypass_loss = ['metric_learning'] # options: 'metric_learning', 'cross_entropy'
        
        # Optimization parameters
        self.loss_ML = 'NTXentLoss' # 'NTXentLoss', 'compose'
        self.miner = None #options: None, 'BatchEasyHardMiner', 'UniformHistogramMiner', 'BatchHardMiner'
        self.t_per_anchor = 16 # triplets per anchor for computing loss if miner is None  
        self.reducer = None #options: None, 'PerAnchorReducer', 'AvgNonZeroReducer'
        self.classifier_optimizer = 'Adam'
        self.lr_scheduler = None #'plateau'
        self.learning_rate = 1e-4  * self.num_devices # 1e-4 = 0.0001
        self.knn_neighbors = 2
        self.knn_test_size = 0.8 # Not used for sequence/video, only for still image
        self.seq_len = 3 # length of the training sequence
        self.step = 5
        
        # Dataset
        self.data_root = os.path.abspath(args.data_root)
        self.drop_last= True if self.num_devices > 1 else False        
        self.num_workers = 2 * self.num_devices
        
        # Miscellaneous
        self.early_test = True # Test before training
        self.tensorboard = False
        self.save_mask = [] # options: 'train', 'test'
        self.nkc = 0.5
        self.r = 0.5
        self.test_dataset = 'davis' # 'davis', 'youtube'
        self.fake = False
        


class ConfigForMSRA10K(DefaultConfig):
    def __init__(self, args, test=True):
        DefaultConfig.__init__(self, args)

        # Meta-data
        self.experiment_name = args.exp
        self.model = 'my_net'
        self.save_model_path = os.path.join('./saved_models/', self.experiment_name)
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())

        # Train/Test setup
        self.epoch = 200
        self.train_batch_size = 1# self.num_devices * 4
        self.test_batch_size = 1#self.num_devices * 4        
        self.train_img_size= (256,256) #(256,256) #(384,384)
        self.val_img_size=(256,256) #(384,384)
        self.kmeans_max_clusters = 30
        
        # MSRA configurations
        self.msra_root = os.path.join(self.data_root, 'MSRA', 'MSRA10K_Imgs_GT')
        self.msra_images = os.path.join(self.msra_root, 'Imgs')
        self.msra_masks = os.path.join(self.msra_root, 'saliency')        
        self.msra_train_annotation = os.path.join(self.msra_root, 'train.txt')
        self.msra_val_annotation = os.path.join(self.msra_root, 'val.txt')        
        self.msra_mean = [0.4532, 0.4335, 0.3921]
        self.msra_std = [0.3025, 0.2918, 0.3122]
        self.normalize = True
        self.do_augmentation = True
        self.n_attempts_augment = 3
        self.split_data = 3
        
        # Superpixels
        self.spx_method = 'slic' # options: 'isec', 'slic', 'precomp'
        self.spx_dir = None
        self.slic_num = 300 # 300

        # Config for test
        if test:
            if self.test_dataset == 'davis':
                self.config_test = ConfigForDAVIS(args, test=False) # None to use the same as for training
            elif self.test_dataset == 'youtube':
                self.config_test = ConfigForYOUTUBE(args, test=False)
        else:
            self.config_test = None
        

class ConfigForDAVIS(DefaultConfig):
    def __init__(self, args, test=True):
        DefaultConfig.__init__(self, args)

        # Meta-data
        self.experiment_name = args.exp
        self.model = 'my_net'
        self.save_model_path = os.path.join('./saved_models/', self.experiment_name)
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        
        # Train/Test setup
        self.epoch = 200
        self.train_batch_size = 1 #self.num_devices * 4
        self.test_batch_size = 1 #self.num_devices * 4
        self.train_img_size=(256,256) #(256,256) # None to keep the original size
        self.val_img_size= None #(256,256) #(384,384) None to keep the original size
        self.kmeans_max_clusters = 10
        
        # DAVIS configurations
        self.version = '2017' # '2016', '2017'
        self.davis_root = os.path.join(self.data_root, 'DAVIS2017')
        self.davis_images = os.path.join(self.davis_root, 'JPEGImages', '480p')
        self.davis_masks = os.path.join(self.davis_root, 'Annotations', '480p')        
        self.davis_train_annotation = os.path.join(self.davis_root, 'ImageSets', '2017', 'train.txt')
        self.davis_val_annotation = os.path.join(self.davis_root, 'ImageSets', '2017', 'val.txt')
        self.davis_mean = [0.4512, 0.4448, 0.4111]
        self.davis_std = [0.2485, 0.2403, 0.2617]
        self.normalize = True
        self.do_augmentation = False
        self.to_return = ['video'] # options: 'frame', '1st_frame', 'video'
        if self.version == '2016':
            self.davis_masks = os.path.join(self.davis_root, 'Annotations', '480p_2016')
            self.davis_val_annotation = os.path.join(self.davis_root, 'ImageSets', '2016', 'val.txt')
            
        
        # Superpixels
        self.spx_method = 'precomp' # options: 'isec', 'slic', 'precomp'
        self.spx_dir = os.path.join(self.davis_root, 'superpixels', 'isec_masks', '480p') # '480p', '384x384'
        #self.spx_dir = os.path.join(self.davis_root, 'superpixels', 'slic_masks', '480p') # '480p', '384x384'
        self.slic_num = 5000 #750 #1500
        
        self.spx_method_train = 'slic' # options: 'isec', 'slic', 'precomp'
        self.slic_num_train = 260
        
        # Config for test
        self.config_test = None # None to use the same as for training
        
        
         # Config for test
        if test:
            if self.test_dataset == 'davis':
                self.config_test = ConfigForDAVIS(args, test=False) # None to use the same as for training
            elif self.test_dataset == 'youtube':
                self.config_test = ConfigForYOUTUBE(args, test=False)
        else:
            self.config_test = None

class ConfigForYOUTUBE(DefaultConfig):
    def __init__(self, args):
        DefaultConfig.__init__(self, args)

        # Meta-data
        self.experiment_name = args.exp
        self.model = 'my_net'
        self.save_model_path = os.path.join('./saved_models/', self.experiment_name)
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        
        # Train/Test setup
        self.epoch = 200
        self.train_batch_size = 1 #self.num_devices * 4
        self.test_batch_size = 1 #self.num_devices * 4
        self.train_img_size=(384,384) #(256,256) # None to keep the original size
        self.val_img_size= 720 #(384,384) #(256,256) # None to keep the original size
        self.kmeans_max_clusters = 10
        
        # DAVIS configurations
        self.yt_imset = 'valid' # 'train', 'test', 'valid'
        self.yt_root = os.path.join(self.data_root, 'YouTubeVOS')
        self.yt_images = os.path.join(self.yt_root, self.yt_imset, 'JPEGImages')
        self.yt_masks = os.path.join(self.yt_root, self.yt_imset, 'Annotations')        
        self.yt_train_annotation = os.path.join(self.yt_root, 'train', 'meta.json')
        self.yt_val_annotation = os.path.join(self.yt_root, 'valid', 'meta.json')
        self.yt_mean = [0.4512, 0.4448, 0.4111]
        self.yt_std = [0.2485, 0.2403, 0.2617]
        self.normalize = True
        self.do_augmentation = False
        self.to_return = ['video'] # options: 'frame', '1st_frame', 'video'
        
        # Superpixels
        self.spx_method = 'precomp' # options: 'isec', 'slic', 'precomp'
        self.spx_dir = os.path.join(self.yt_root, self.yt_imset, 'superpixels', 'isec_masks', '720p') # '480p', '384x384'
        #self.spx_dir = os.path.join(self.yt_root, self.yt_imset, 'superpixels', 'slic_masks', '384x384') # '480p', '384x384'
        self.slic_num = 375 #750 #1500
        self.save_spx = False
        
        # Config for test
        self.config_test = None # None to use the same as for training

################################################
def init(args):
    config = None
    if args.dataset.lower() in ['msra10k', 'msra']:
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
        cfg_dict = vars(config)
        for k, v in cfg_dict.items():
            f.write('{}: {}\n'.format(k,v))
            #print('{}: {}'.format(k,v))        
        if config.config_test:
            f.write('-------config for test dataset:\n')
            for k, v in vars(config.config_test).items():
                if not k in cfg_dict.keys() or v != cfg_dict[k]:
                    f.write('{}: {}\n'.format(k,v))
                    #print('{}: {}'.format(k,v))
        f.write('\n')
    

def set_config(jup_notebook=False, dataset='MSRA10k'):
    parser = argparse.ArgumentParser(description='My_Net Training') 
    parser.add_argument("--dataset", type=str, default='MSRA10K')
    parser.add_argument("--exp", type=str, default='')
    parser.add_argument("--resume_model_path", type=str, default='')
    parser.add_argument('--gpu_id', type=str, default='', help='gpu id')
    parser.add_argument("--data_root", type=str, default='../databases', help='the dir of dataset')
    parser.add_argument("--reset_iou", type=bool, default=False)
    
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