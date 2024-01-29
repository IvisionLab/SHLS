#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
from tqdm import tqdm
from ..config import set_config
import torchvision.transforms as T


##################### MSRA #######################
class MSRA_only_img(data.Dataset):    
    def __init__(self, config, imset='train'):
        #../databases/MSRA/MSRA_B
        if not os.path.isdir(config.msra_root):
            raise RuntimeError('Dataset not found: {}'.format(config.msra_root))
        
        self.imset = imset 
        self.image_dir = config.msra_images
        
        self.img_transf = T.Compose([T.ToTensor(),
                                     T.Normalize(mean=[0.4512, 0.4448, 0.4111],
                                                 std=[0.2485, 0.2403, 0.2617])])
        
        if self.imset == 'train':
            self.img_size = config.train_img_size
            split_f = config.msra_train_annotation
        elif self.imset == 'test':
            self.img_size = config.val_img_size
            split_f = config.msra_val_annotation
        else:
            raise RuntimeError('Dataset split \'{}\' not found'.format(self.imset))
        
        ext = None if config.dataset.lower() == 'msra10k' else -4            
        
        with open(split_f, "r") as f:
            self.img_list = [x.strip()[:ext] for x in f.readlines()]
        
    
    def __getitem__(self, idx):
                
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")     
        
        img = np.array(Image.open(img_path).convert('RGB'), dtype="float32")/255.
        
        img = cv2.resize(np.copy(img), self.img_size)      
                    
        #img = torch.from_numpy(img).permute(2,0,1)
        
        img = self.img_transf(img)
            
        return img

    def __len__(self):
        return len(self.img_list)

##################### DAVIS ######################
class DAVIS_only_img(data.Dataset):    
    def __init__(self, config, imset='train'):
        if not os.path.isdir(config.davis_root):
            raise RuntimeError('Dataset not found: {}'.format(config.davis_root))
        
        self.imset = imset 
        self.image_dir = config.davis_images
        
        self.img_transf = T.Compose([T.ToTensor(),
                                     T.Normalize(mean=[0.4532, 0.4335, 0.3921],
                                                 std=[0.3025, 0.2918, 0.3122])])
        
        self.frames = {}
        self.num_frames = {}
        
        if self.imset == 'train':
            self.img_size = config.train_img_size
            _imset_f = config.davis_train_annotation
        else:
            self.img_size = config.val_img_size
            _imset_f = config.davis_val_annotation
        
        with open(os.path.join(_imset_f), "r") as lines:
            idx = 0
            for line in lines:
                _video = line.rstrip('\n')                
                for f_num, f in enumerate(os.listdir(os.path.join(self.image_dir, _video))):
                    if f.endswith(".jpg"):
                        self.frames[idx] = ('{:05d}'.format(f_num), _video)
                        idx += 1
                    #if imset != 'train' and f_num > 9:
                     #   break
                self.num_frames[_video] = f_num + 1
    
    def __getitem__(self, index):
        video = self.frames[index][1]
        
        img_path = os.path.join(self.image_dir, video, self.frames[index][0] + '.jpg')
        
        img = np.array(Image.open(img_path).convert('RGB'), dtype="float32")/255.
        
        img = cv2.resize(np.copy(img), self.img_size)
        
        #img = torch.from_numpy(img).permute(2,0,1)
        
        img = self.img_transf(img)
        
        return img
            
    
    def __len__(self):
        return len(self.frames)


def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    img_max = 0.0
    img_min = 99999999.9
    for images in tqdm(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        
        img_max = images.max() if images.max() > img_max else img_max
        img_min = images.min() if images.min() < img_min else img_min

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    print('max: {}, min: {}'.format(img_max, img_min))    
    return mean,std


if __name__ == '__main__':
    
    imset = 'train'
    config = set_config() 
    config.do_augmentation = False
    if config.dataset.lower() == 'msra10k':
        data_set = MSRA_only_img(config, imset=imset)
    elif config.dataset.lower() == 'davis':
        data_set = DAVIS_only_img(config, imset=imset)
    else:
        raise RuntimeError('Dataset not found: {}'.format(config.dataset))
        
    train_loader = data.DataLoader(data_set, batch_size=config.train_batch_size, 
                                       shuffle=False, num_workers=config.num_workers, drop_last=False)
    
    mean, std = batch_mean_and_sd(train_loader)
    print('Dataset: {}, img_set: {}, img_size: {}'.format(config.dataset, imset, config.train_img_size))
    print("mean and std: \n", mean, std)

