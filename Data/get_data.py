#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import sys
#sys.path.append('./Data/')
import torch
import os
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
from skimage.measure import label
from isec_py.isec import isec, shell_kernel, remove_borders
from Data.augmentation import Data_Augmentation, remove_small_objcts
from Data.aug_davis import aug_heavy

   
##################### MSRA #######################
class MSRA_dataset(data.Dataset):    
    def __init__(self, config, imset='train'):
        #../databases/MSRA/MSRA_B
        if not os.path.isdir(config.msra_root):
            raise RuntimeError('Dataset not found: {}'.format(config.msra_root))
        
        self.imset = imset 
        self.image_dir = config.msra_images
        self.mask_dir = config.msra_masks        
        self.pre_computed_spx = config.pre_computed_spx
        self.do_augmentation = config.do_augmentation
        
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
        
        ext = None if config.dataset.lower() == 'msra10k' else -4            
        
        with open(split_f, "r") as f:
            self.img_list = [x.strip()[:ext] for x in f.readlines()]
        
        if self.do_augmentation:
            self.augmentation = Data_Augmentation(config, self.img_list)
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")        
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = (np.array(Image.open(mask_path).convert('P')) > 0).astype(np.uint8)
        
        img = cv2.resize(np.copy(img), self.img_size)
        mask = cv2.resize(np.copy(mask), self.img_size, interpolation=cv2.INTER_NEAREST)
           
        obj_contour, obj_kernel = shell_kernel(mask)
        #obj_contour, _ = shell_kernel(obj_kernel)
        obj_label = remove_borders(label((1-obj_contour).astype(np.int32), connectivity=1, return_num=False))
        
        if self.do_augmentation:
            img, obj_label, num_obj = self.augmentation.insert_random_objects(img, obj_label, idx=idx)
        else:
            obj_label, num_obj = remove_small_objcts(obj_label)
                                                                
        
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

##################### DAVIS ######################
class DAVIS_dataset(data.Dataset):    
    def __init__(self, config, imset='train'):
        if not os.path.isdir(config.davis_root):
            raise RuntimeError('Dataset not found: {}'.format(config.davis_root))
        
        self.imset = imset 
        self.image_dir = config.davis_images
        self.mask_dir = config.davis_masks        
        self.pre_computed_spx = config.pre_computed_spx
        self.max_img_side = config.max_img_side
        self.do_augmentation = config.do_augmentation
        
        if not self.pre_computed_spx:
            self.isec = isec(f_order=11, nit=4)
        else:
            self.spx_dir = os.path.join(config.spx_dir)
        
        self.frames = {}
        self.num_frames = {}                
        
        _imset_f = config.davis_train_annotation if imset == 'train' else config.davis_val_annotation        
        with open(os.path.join(_imset_f), "r") as lines:
            idx = 0
            for line in lines:
                _video = line.rstrip('\n')                
                for f_num, f in enumerate(os.listdir(os.path.join(self.image_dir, _video))):
                    if f.endswith(".jpg"):
                        self.frames[idx] = ('{:05d}'.format(f_num), _video)
                        idx += 1
                self.num_frames[_video] = f_num + 1
       
        if self.do_augmentation:
            self.augmentation = aug_heavy()
    
    def __getitem__(self, index):
        #index += 69+250
        video = self.frames[index][1]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['frame_idx'] = int(self.frames[index][0])
        
        img_path = os.path.join(self.image_dir, video, self.frames[index][0] + '.jpg')
        mask_path = os.path.join(self.mask_dir, video, self.frames[index][0] + '.png') 
        
        img = np.array(Image.open(img_path).convert('RGB'))#/255.
        mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
        
        if self.do_augmentation:
            frame_list, mask_list = self.augmentation([img],[mask])            
            img = frame_list[0]
            mask = mask_list[0]
        else:
            img = self.resize_keeping_aspect_ratio(img/255., max_size=self.max_img_side)
            mask = self.resize_keeping_aspect_ratio(mask, max_size=self.max_img_side)
        
        obj_label = (mask + 1).astype(np.int32) # first label (background) set to 1
        num_obj = obj_label.max()
        
        if not self.pre_computed_spx:
            spx, _ = self.isec.segment(img)
        else:
            spx_path = os.path.join(self.spx_dir, video, self.frames[index][0] + '.png')
            spx = np.array(Image.open(spx_path))

        img = torch.from_numpy(img).permute(2,0,1)
        obj_label = torch.from_numpy(obj_label).unsqueeze(0)        
        spx = torch.from_numpy(spx).unsqueeze(0)
        
        return img, spx, obj_label, num_obj, info
            
    
    def __len__(self):
        return len(self.frames)
    
    
    def resize_keeping_aspect_ratio(self, img, max_size=480, enlarge=False):   
        img_w, img_h = img.shape[0], img.shape[1]
        
        if max_size is None:
            return img
    
        if not enlarge and img_w <= max_size and img_h <= max_size:
            return img
        
        if img_w >= img_h:
            wsize = max_size
            hsize = int((float(img_h)*float(wsize/float(img_w))))        
        else:
            hsize = max_size
            wsize = int((float(img_w)*float(hsize/float(img_h))))
            
        if len(img.shape) > 2 and img.shape[2] > 1:
            return cv2.resize(img, (hsize,wsize))
        else:
            return cv2.resize(img, (hsize,wsize), interpolation=cv2.INTER_NEAREST)


   
##################################################
def get_data(config, rank=None, world_size=None):
    train_loader = None
    test_loader = None
    
    if config.dataset.lower() in ['msra10k', 'msra_b', 'davis']:
        
        if config.dataset.lower() == 'davis':
            train_set = DAVIS_dataset(config, imset='train')
            test_set = DAVIS_dataset(config, imset='test')
        else:
            train_set = MSRA_dataset(config, imset='train')
            test_set = MSRA_dataset(config, imset='test')
        
        train_loader = data.DataLoader(train_set, batch_size=config.train_batch_size, 
                                       shuffle=True, num_workers=config.num_workers, drop_last=config.drop_last,
                                       pin_memory=True)
        test_loader = data.DataLoader(test_set, batch_size=config.test_batch_size, 
                                       shuffle=True, num_workers=config.num_workers, drop_last=config.drop_last,
                                       pin_memory=True)
    else:
        raise RuntimeError('Incorrect dataset name: {}'.format(config.dataset))
    
    return train_loader, test_loader

def get_data_ddp(config, rank, world_size):
    train_loader = None
    test_loader = None
    
    if config.dataset.lower() in ['msra10k', 'msra_b', 'davis']:
        from torch.utils.data.distributed import DistributedSampler
        
        if config.dataset.lower() == 'davis':
            train_set = DAVIS_dataset(config, imset='train')
            test_set = DAVIS_dataset(config, imset='test')
        else:
            train_set = MSRA_dataset(config, imset='train')
            test_set = MSRA_dataset(config, imset='test')
                        
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, 
                                           drop_last=config.drop_last)
        train_loader = data.DataLoader(train_set, batch_size=config.train_batch_size, shuffle=False,
                                       num_workers=0, drop_last=config.drop_last, pin_memory=False, sampler=train_sampler)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=True, 
                                           drop_last=config.drop_last)
        test_loader = data.DataLoader(test_set, batch_size=config.train_batch_size, shuffle=False,
                                       num_workers=0, drop_last=config.drop_last, pin_memory=False, sampler=test_sampler)        
    else:
        raise RuntimeError('Incorrect dataset name: {}'.format(config.dataset))
    
    return train_loader, test_loader


if __name__ == "__main__":    
    
    print('Testing get_data.py')           
    
    
    
    