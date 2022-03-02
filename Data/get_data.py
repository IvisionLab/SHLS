#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import sys
#sys.path.append('./Data/')
import torch
import os.path
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
from skimage.measure import label
from isec_py.isec import isec, shell_kernel, remove_borders


##################### MSRA10K #######################
class MSRA10K_dataset(data.Dataset):    
    def __init__(self, config, imset='train'):
        #../databases/MSRA/MSRA10K_Imgs_GT
        if not os.path.isdir(config.msra_root):
            raise RuntimeError('Dataset not found: {}'.format(config.msra_root))
        
        self.imset = imset
        self.image_dir = config.msra_images
        self.mask_dir = config.msra_masks
        self.train_img_size = config.train_img_size
        self.pre_computed_spx = config.pre_computed_spx
        if not self.pre_computed_spx:
            self.isec = isec(nit=4)
        else:
            self.spx_dir = config.spx_dir
        
        self.img_list = []
        for file in os.listdir(self.image_dir):
            if file.endswith(".jpg"):
                self.img_list.append(file.rstrip('.jpg'))
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
        mask = (mask/255).astype(np.uint8)
        
        img = cv2.resize(np.copy(img), self.train_img_size)
        mask = cv2.resize(np.copy(mask), self.train_img_size, interpolation=cv2.INTER_NEAREST)
        
           
        obj_contour, _ = shell_kernel(mask)
        obj_label = remove_borders(label((1-obj_contour).astype(int), connectivity=1, return_num=False))
        num_obj = np.max(obj_label)
                                                                
        
        if not self.pre_computed_spx:
            spx, _ = self.isec.segment(img)
        else:
            spx_path = os.path.join(self.spx_dir, self.img_list[idx] + ".png")
            spx = np.array(Image.open(spx_path).convert('P'), dtype=np.uint8)
            
        
        img = torch.from_numpy(img).permute(2,0,1)
        obj_label = torch.from_numpy(obj_label).unsqueeze(0)        
        spx = torch.from_numpy(spx).unsqueeze(0)
      
        info = {}
        info['name'] = self.img_list[idx]
        
        return img, spx, obj_label, num_obj, info

    def __len__(self):
        return len(self.img_list)
    
##################### MSRA_B #######################
class MSRA_B_dataset(data.Dataset):    
    def __init__(self, config, imset='train'):
        #../databases/MSRA/MSRA_B
        if not os.path.isdir(config.msra_root):
            raise RuntimeError('Dataset not found: {}'.format(config.msra_root))
        
        self.imset = imset 
        self.image_dir = config.msra_images
        self.mask_dir = config.msra_masks        
        self.pre_computed_spx = config.pre_computed_spx
        
        if self.imset == 'train':
            self.img_size = config.train_img_size
            split_f = config.msra_train_annotation
        elif self.imset == 'test':
            self.img_size = config.val_img_size
            split_f = config.msra_val_annotation
        else:
            raise RuntimeError('Dataset split \'{}\' not found'.format(self.imset))
        
        if not self.pre_computed_spx:
            self.isec = isec(nit=4)
        else:
            self.spx_dir = os.path.join(config.spx_dir,
                                        str(self.img_size[0])+'x'+str(self.img_size[1]))
        
        with open(split_f, "r") as f:
            self.img_list = [x.strip()[:-4] for x in f.readlines()]
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        
        mask_path = [os.path.join(x, self.img_list[idx] + ".png") for x in self.mask_dir]
        
        img = np.array(Image.open(img_path).convert('RGB'))
        
        masks = [np.array(Image.open(x).convert('P'), dtype=np.uint8) for x in mask_path]
        masks = [x/255 for x in masks]
        
        mask = masks[0][:,:,None]        
        for i in range(1,len(masks)):
            mask = np.concatenate((mask, masks[i][:, :, None]), axis=2)
        mask = (np.sum(mask, axis=(2)) > 0).astype(np.uint8)     
        
        img = cv2.resize(np.copy(img), self.img_size)
        mask = cv2.resize(np.copy(mask), self.img_size, interpolation=cv2.INTER_NEAREST)
           
        obj_contour, obj_kernel = shell_kernel(mask)
        #obj_contour, _ = shell_kernel(obj_kernel)
        obj_label = remove_borders(label((1-obj_contour).astype(int), connectivity=1, return_num=False))
        num_obj = np.max(obj_label)
                                                                
        
        if not self.pre_computed_spx:
            spx, _ = self.isec.segment(img)
        else:
            spx_path = os.path.join(self.spx_dir, self.img_list[idx] + ".png")
            spx = np.array(Image.open(spx_path))
            
            
        img = torch.from_numpy(img).permute(2,0,1)
        obj_label = torch.from_numpy(obj_label).unsqueeze(0)        
        spx = torch.from_numpy(spx).unsqueeze(0)
         
        info = {}
        info['name'] = self.img_list[idx]
        
        return img, spx, obj_label, num_obj, info

    def __len__(self):
        return len(self.img_list)


def get_data(config):
    train_loader = None
    test_loader = None
    if config.dataset.lower() == 'msra10k':
        train_loader = data.DataLoader(MSRA10K_dataset(config, imset='train'), batch_size=config.train_batch_size, 
                                       shuffle=True, num_workers=config.num_workers)
    elif config.dataset.lower() in ['msra_b' , 'msra_b_sdumont', 'msra_b_pinha']:
        train_loader = data.DataLoader(MSRA_B_dataset(config, imset='train'), batch_size=config.train_batch_size, 
                                       shuffle=True, num_workers=config.num_workers, drop_last=config.drop_last,
                                       pin_memory=True)
        test_loader = data.DataLoader(MSRA_B_dataset(config, imset='test'), batch_size=config.test_batch_size, 
                                       shuffle=True, num_workers=config.num_workers, drop_last=config.drop_last,
                                       pin_memory=True)
    
    
    return train_loader, test_loader


if __name__ == "__main__":    
    
    print('Testing get_data.py')           
    
    
    
    