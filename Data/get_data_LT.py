#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import sys
#sys.path.append('./Data/')
import torch
import os
import numpy as np
import json
import glob
from PIL import Image
from torch.utils import data
import torchvision.transforms as T
from torch.utils.data.distributed import DistributedSampler
from Data.eval_sampler import DistributedEvalSampler
from skimage.measure import label as to_label
from isec_py.isec import isec, shell_kernel, remove_borders
from Data.augmentation import Data_Augmentation
from Data.aug_davis import aug_heavy
from utils_LT import reorder_labels, to_resize
from skimage.segmentation import slic
import math



##################### MSRA #######################
class MSRA_dataset(data.Dataset):    
    def __init__(self, config, imset='train'):
        if not os.path.isdir(config.msra_root):
            raise RuntimeError('Dataset not found: {}'.format(config.msra_root))
        
        self.imset = imset 
        self.image_dir = config.msra_images
        self.mask_dir = config.msra_masks
        self.spx_method = config.spx_method
        self.img_transf = T.ToTensor()
        if config.normalize:
            self.img_transf = T.Compose([T.ToTensor(),T.Normalize(mean=config.msra_mean,
                                                                  std=config.msra_std)])
        if self.imset == 'train':
            self.img_size = config.train_img_size
            split_f = config.msra_train_annotation
        elif self.imset == 'test':
            self.img_size = config.val_img_size
            split_f = config.msra_val_annotation
        else:
            raise RuntimeError('Dataset split \'{}\' not found'.format(self.imset))
        
        if self.spx_method == 'precomp':
            self.spx_dir = os.path.join(config.spx_dir,
                            str(self.img_size[0])+'x'+str(self.img_size[1]))            
        elif self.spx_method == 'slic':
            self.slic_num = config.slic_num
        else:
            self.isec = isec(f_order=11, nit=4)
                    
        ext = None if config.dataset.lower() == 'msra10k' else -4     
        with open(split_f, "r") as f:
            self.img_list = [x.strip()[:ext] for x in f.readlines()]
        
        self.split_data = config.split_data
        self.aug_obj = Data_Augmentation(config, self.img_list, split_data=self.split_data)
        self.seq_len = config.seq_len
    
    def __getitem__(self, idx):        
        if self.split_data:
            slc = int(len(self.img_list)/self.split_data)
            idx = idx + np.random.choice(list(range(0, slc*self.split_data, slc))+[len(self.img_list)-slc])
        info = {}
        info['name'] = self.img_list[idx]        
        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")        
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")        
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)/255.
        mask = (np.array(Image.open(mask_path).convert('P')) > 0).astype(np.uint8)
        
        #img = cv2.resize(np.copy(img), (self.img_size[1],self.img_size[0]))
        #mask = cv2.resize(np.copy(mask), (self.img_size[1],self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        obj_contour, obj_kernel = shell_kernel(mask)
        label = remove_borders(to_label((1-obj_contour).astype(np.int32), connectivity=1, return_num=False)).astype(np.int32)
        
        img_seq = torch.zeros((self.seq_len, 3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
        spx_seq = torch.zeros((self.seq_len, 1, self.img_size[0], self.img_size[1]), dtype=torch.int32)
        label_seq = torch.zeros((self.seq_len, 1, self.img_size[0], self.img_size[1]), dtype=torch.int32)
        
        np_img_seq, np_label_seq = self.aug_obj.make_sequence(img, label, idx=idx, seq_len=self.seq_len)
        
        for t in range(self.seq_len):
            
            if self.spx_method == 'precomp':
                spx_path = os.path.join(self.spx_dir, self.img_list[idx] + ".png")
                spx = np.array(Image.open(spx_path), dtype=np.int32)            
            elif self.spx_method == 'slic':
                spx = slic(np_img_seq[t], n_segments=self.slic_num, compactness=20, 
                                     sigma=1, multichannel=True, convert2lab=True) + 1
            else:
                spx, _ = self.isec.segment(np_img_seq[t])            
                    
            #img_seq[t] = torch.from_numpy(np_img_seq[t]).permute(2,0,1)
            img_seq[t] = self.img_transf(np_img_seq[t])
            label_seq[t] = reorder_labels(torch.from_numpy(np_label_seq[t])).unsqueeze(0)   
            label_seq[t, label_seq[t]==0] = 1 # pass to background the black pixels caused by img rotation
            spx_seq[t] = torch.from_numpy(spx).unsqueeze(0)
        
        num_cls = label_seq[0].max()
        
        return img_seq, spx_seq, label_seq, num_cls, info

    def __len__(self):
        if self.split_data: return int(len(self.img_list)/self.split_data)
        return len(self.img_list)

##################### DAVIS Train #######################
class DAVIS_train_dataset(data.Dataset):    
    def __init__(self, config, imset='train'):
        if not os.path.isdir(config.davis_root):
            raise RuntimeError('Dataset not found: {}'.format(config.davis_root))
        
        self.imset = imset 
        self.image_dir = config.davis_images
        self.mask_dir = config.davis_masks        
        self.spx_method = config.spx_method_train
        self.do_augmentation = config.do_augmentation
        self.seq_len = config.seq_len
        self.step = config.step
        self.img_transf = T.ToTensor()
        if config.normalize:
            self.img_transf = T.Compose([T.ToTensor(),T.Normalize(mean=config.davis_mean,
                                                                  std=config.davis_std)])
        
        if self.spx_method == 'precomp':
            self.spx_dir = os.path.join(config.spx_dir)            
        elif self.spx_method == 'slic':
            self.slic_num = config.slic_num_train
        else:
            self.isec = isec(f_order=11, nit=4)
        
        
        self.seq_frames = {}
        self.seq_name = {}
        self.num_classes = {}
        self.lbl_map = {}
        
        if self.imset == 'train':
            self.img_size = config.train_img_size
            _imset_f = config.davis_train_annotation
        else:
            self.img_size = config.val_img_size
            _imset_f = config.davis_val_annotation
        
        with open(os.path.join(_imset_f), "r") as lines:
            idx = 0
            for line in lines:
                seq_name = line.rstrip('\n')
                if seq_name not in self.num_classes.keys():
                    _mask = np.array(Image.open(os.path.join(self.mask_dir, seq_name, '00000.png')).convert("P"))
                    _mask = to_resize(_mask, new_size=self.img_size)
                    _mask = (_mask + 1).astype(np.int32)
                    _, _num_cls, _lbl_map =  map_labels(_mask, lbl_map=None)
                    self.lbl_map[seq_name] = _lbl_map
                    self.num_classes[seq_name] = _num_cls
                if self.num_classes[seq_name] < 2:
                    continue                    
                _all_frames = os.listdir(os.path.join(self.image_dir, seq_name))                
                _all_frames = [int(os.path.basename(x)[:-4]) for x in _all_frames if x.endswith(".jpg")]
                _all_frames.sort()
                n_seq = len(_all_frames) - (self.step*(self.seq_len-1))  
                for f_num in range(1, n_seq):                    
                    f = _all_frames[:1] + _all_frames[f_num::self.step][:self.seq_len-1]
                    self.seq_frames[idx] = f
                    self.seq_name[idx] = seq_name
                    idx += 1
       
        if self.do_augmentation:
            crop_size = max(self.img_size) if isinstance(self.img_size, tuple) else self.img_size
            self.augmentation = aug_heavy(crop_size)
        
    
    def __getitem__(self, idx):
        
        seq_name = self.seq_name[idx]
        info = {}
        info['name'] = seq_name
        info['num_frames'] = self.seq_len
        seq_len = self.seq_len
        frames = self.seq_frames[idx]
        h, w = self.img_size
        num_cls = self.num_classes[seq_name]
        lbl_map = self.lbl_map[seq_name]
        info['lbl_map'] = lbl_map
        info['frames'] = frames
        
        
        img_seq = torch.zeros((seq_len, 3, h, w), dtype=torch.float32)
        spx_seq = torch.zeros((seq_len, 1, h, w), dtype=torch.int32)
        label_seq = torch.zeros((seq_len, 1, h, w), dtype=torch.int32)
        
        for n in range(seq_len):
            
            f_num = frames[n]
            img_path = os.path.join(self.image_dir, seq_name, '{:05d}.jpg'.format(f_num))
            mask_path = os.path.join(self.mask_dir, seq_name, '{:05d}.png'.format(f_num))
        
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)/255.
            mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
            
            img = to_resize(img, new_size=self.img_size)
            mask = to_resize(mask, new_size=self.img_size)       
            label = (mask + 1).astype(np.int32) # first label (background) set to 1
            
            if self.spx_method == 'precomp':
                spx_path = os.path.join(self.spx_dir, seq_name, '{:05d}.png'.format(f_num))
                spx = np.array(Image.open(spx_path), dtype=np.int32)                            
            elif self.spx_method == 'slic':
                spx = slic(img, n_segments=self.slic_num, compactness=20, 
                           sigma=1, multichannel=True, convert2lab=True) + 1
            else:
                spx, _ = self.isec.segment(img)
        
            #img = torch.from_numpy(img).permute(2,0,1)
            img_seq[n] = self.img_transf(img)
            spx_seq[n] = torch.from_numpy(spx).unsqueeze(0)
            #label_seq[n] = reorder_labels(torch.from_numpy(label)).unsqueeze(0)
            _label, _, _ = map_labels(label, lbl_map)
            label_seq[n] = _label.unsqueeze(0)
        
        return img_seq, spx_seq, label_seq, num_cls, info


    def __len__(self):
        return len(self.seq_frames)

################## DAVIS (Frames) ####################
class DAVIS_frame_dataset(data.Dataset):    
    def __init__(self, config, imset='train'):
        if not os.path.isdir(config.davis_root):
            raise RuntimeError('Dataset not found: {}'.format(config.davis_root))
        
        self.imset = imset 
        self.image_dir = config.davis_images
        self.mask_dir = config.davis_masks        
        self.spx_method = config.spx_method
        self.do_augmentation = config.do_augmentation
        self.return_1st_frame = '1st_frame' in config.to_return
        self.img_transf = T.ToTensor()
        if config.normalize:
            self.img_transf = T.Compose([T.ToTensor(),T.Normalize(mean=config.davis_mean,
                                                                  std=config.davis_std)])
        
        if self.spx_method == 'precomp':
            self.spx_dir = os.path.join(config.spx_dir)            
        elif self.spx_method == 'slic':
            self.slic_num = config.slic_num
        else:
            self.isec = isec(f_order=11, nit=4)
        
        
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
                seq_name = line.rstrip('\n')                
                for f_num, f in enumerate(os.listdir(os.path.join(self.image_dir, seq_name))):
                    if f.endswith(".jpg"):
                        self.frames[idx] = ('{:05d}'.format(f_num), seq_name)
                        idx += 1
                    if imset != 'train' and f_num > 9:
                        break
                self.num_frames[seq_name] = f_num + 1
       
        if self.do_augmentation:
            crop_size = max(self.img_size) if isinstance(self.img_size, tuple) else self.img_size
            self.augmentation = aug_heavy(crop_size)
        
        self.first_itens = {}

    
    def __getitem__(self, index):
        #index += 69+50+80+84+90+75#+40+104+90+60+66+52
        seq_name = self.frames[index][1]
        info = {}
        info['name'] = seq_name
        info['num_frames'] = self.num_frames[seq_name]
        info['frame_idx'] = int(self.frames[index][0])
        
        img_path = os.path.join(self.image_dir, seq_name, self.frames[index][0] + '.jpg')
        mask_path = os.path.join(self.mask_dir, seq_name, self.frames[index][0] + '.png') 
        
        
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)/255.
        mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
       
        if self.do_augmentation and self.imset == 'train':
            frame_list, mask_list = self.augmentation([img],[mask])            
            img = frame_list[0]
            mask = mask_list[0]
        else:
            img = to_resize(img, new_size=self.img_size)
            mask = to_resize(mask, new_size=self.img_size)
        
        label = (mask + 1).astype(np.int32) # first label (background) set to 1
        
        if self.spx_method == 'precomp':
            spx_path = os.path.join(self.spx_dir, seq_name, self.frames[index][0] + '.png')
            spx = np.array(Image.open(spx_path), dtype=np.int32)                            
        elif self.spx_method == 'slic':
            spx = slic(img, n_segments=self.slic_num, compactness=20, 
                       sigma=1, multichannel=True, convert2lab=True) + 1
        else:
            spx, _ = self.isec.segment(img)
        
        #img = torch.from_numpy(img).permute(2,0,1)
        img = self.img_transf(img)
        spx = torch.from_numpy(spx).unsqueeze(0)
        label = reorder_labels(torch.from_numpy(label)).unsqueeze(0) 
        num_cls = label.max()
        
        if self.return_1st_frame:
            if info['frame_idx'] == 0:
                self.first_itens[seq_name] = [img, spx, label]             
            return  (img, spx, label, num_cls, info, *self.first_itens[seq_name])
        
        return img, spx, label, num_cls, info
            
    def __len__(self):
        return len(self.frames)
        
    
################## DAVIS (Videos) ####################
class DAVIS_video_dataset(data.Dataset):    
    def __init__(self, config, imset='train'):
        if not os.path.isdir(config.davis_root):
            raise RuntimeError('Dataset not found: {}'.format(config.davis_root))
        
        self.imset = imset
        self._2016 = config.version == '2016'
        self.image_dir = config.davis_images
        self.mask_dir = config.davis_masks        
        self.spx_method = config.spx_method
        self.img_transf = T.ToTensor()
        if config.normalize:
            self.img_transf = T.Compose([T.ToTensor(),T.Normalize(mean=config.davis_mean,
                                                                  std=config.davis_std)])
        if self.spx_method == 'precomp':
            self.spx_dir = os.path.join(config.spx_dir)            
        elif self.spx_method == 'slic':
            self.slic_num = config.slic_num
        else:
            self.isec = isec(f_order=11, nit=4) #isec(f_order=11, nit=4)
            #self.isec = isec(f_order=config.nkc, nit=5)
        
        if self.imset == 'train':
            self.img_size = config.train_img_size
            _imset_f = config.davis_train_annotation
        else:
            self.img_size = config.val_img_size
            _imset_f = config.davis_val_annotation
        
        self.videos = []
        self.num_frames = {}
        self.max_num_frames = 1000
        self.num_classes = {}
        self.shape = {}
        self.orig_shape = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                seq_name = line.rstrip('\n')
                self.videos.append(seq_name)
                self.num_frames[seq_name] = min(len(os.listdir(os.path.join(self.image_dir, seq_name))), self.max_num_frames)                
                _mask = np.array(Image.open(os.path.join(self.mask_dir, seq_name, '00000.png')).convert("P"))
                if self._2016: _mask = (_mask==255).astype(np.int8)
                self.num_classes[seq_name] = np.max(_mask) +1
                self.orig_shape[seq_name] = np.shape(_mask)
                _mask = to_resize(_mask, new_size=self.img_size)
                self.shape[seq_name] = np.shape(_mask)
                
    
    def __getitem__(self, index):
        seq_name = self.videos[index]
        info = {}
        info['name'] = seq_name
        seq_len = self.num_frames[seq_name]
        info['num_frames'] = seq_len
        info['orig_shape'] = self.orig_shape[seq_name]
        num_cls = self.num_classes[seq_name]        
        h, w = self.shape[seq_name]
        
        img_seq = torch.zeros((seq_len, 3, h, w), dtype=torch.float32)
        spx_seq = torch.zeros((seq_len, 1, h, w), dtype=torch.int32)
        label_seq = torch.zeros((seq_len, 1, h, w), dtype=torch.int32)
        
        for n in range(seq_len):
            
            img_path = os.path.join(self.image_dir, seq_name, '{:05d}.jpg'.format(n))
            mask_path = os.path.join(self.mask_dir, seq_name, '{:05d}.png'.format(n))
        
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)/255.
            mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
            if self._2016: mask = (mask==255).astype(np.int8)
            
            img = to_resize(img, new_size=self.img_size)
            mask = to_resize(mask, new_size=self.img_size)       
            label = (mask + 1).astype(np.int32) # first label (background) set to 1
            
            if self.spx_method == 'precomp':
                spx_path = os.path.join(self.spx_dir, seq_name, '{:05d}.png'.format(n))
                spx = np.array(Image.open(spx_path), dtype=np.int32)                            
            elif self.spx_method == 'slic':
                spx = slic(img, n_segments=self.slic_num, compactness=20, 
                           sigma=1, multichannel=True, convert2lab=True) + 1
            elif self.spx_method == 'grid':
                spx = self.make_grid(h, w, 1024)
            else:
                spx, _ = self.isec.segment(img)
        
            #img = torch.from_numpy(img).permute(2,0,1)
            img_seq[n] = self.img_transf(img)
            spx_seq[n] = torch.from_numpy(spx).unsqueeze(0)
            #label_seq[n] = reorder_labels(torch.from_numpy(label)).unsqueeze(0)
            label_seq[n] = torch.from_numpy(label).unsqueeze(0)
        
        return img_seq, spx_seq, label_seq, num_cls, info
            
    def __len__(self):
        return len(self.videos)
    
    def make_grid(self, height, width, num):    
        BOX_H = int(height/math.sqrt(num))
        BOX_W = int(width/math.sqrt(num))        
        arr = np.empty((height, width), dtype=np.int32)    
        ccy = 0
        ccx = 0
        for y in range(0, height, BOX_H):
            for x in range(0, width, BOX_W):
                arr[y:y+BOX_H, x:x+BOX_W] = ccy + ccx + 1
                ccx += 1
            ccy += 1
            ccx -= 1
        return arr


################## YOUTUBE (Videos) ####################
class YOUTUBE_video_dataset(data.Dataset):    
    def __init__(self, config, imset='valid'):
        if not os.path.isdir(config.yt_root):
            raise RuntimeError('Dataset not found: {}'.format(config.yt_root))
        
        self.cfg = config
        self.imset = config.yt_imset 
        self.image_dir = config.yt_images
        self.mask_dir = config.yt_masks        
        self.spx_method = config.spx_method
        self.img_transf = T.ToTensor()
        self.save_spx = config.save_spx
        if config.normalize:
            self.img_transf = T.Compose([T.ToTensor(),T.Normalize(mean=config.yt_mean,
                                                                  std=config.yt_std)])
        if self.save_spx:
            self.spx_dir = os.path.join(config.spx_dir)
        if self.spx_method == 'precomp':
            self.spx_dir = os.path.join(config.spx_dir)            
        elif self.spx_method == 'slic':
            self.slic_num = config.slic_num
        else:
            self.isec = isec(f_order=11, nit=4)
        
        if self.imset == 'train':
            self.img_size = config.train_img_size
            _imset_f = config.yt_train_annotation
        else:
            self.img_size = config.val_img_size
            _imset_f = config.yt_val_annotation
        
        self.videos = []
        self.num_frames = {}
        self.max_num_frames = 1000
        self.num_cls = {}
        self.orig_cls = {}
        self.shape = {}
        self.orig_shape = {}
        self.ref_masks = {}
        self.frame_idx = {}
        self.all_masks = {}
        self.ref_per_cls = {} # 1st mask of each object (key=obj, value=ref_mask)        
        
        with open(_imset_f) as json_file:
            json_data = json.load(json_file)
            for seq_name in list(json_data['videos'].keys()):
                self.videos.append(seq_name)
                _orig_cls = list(json_data['videos'][seq_name]['objects'].keys())
                self.orig_cls[seq_name] = [int(c) for c in _orig_cls]
                self.num_cls[seq_name] = max(1, len(json_data['videos'][seq_name]['objects']))+1                
                _frame_paths = glob.glob(os.path.join(self.image_dir, seq_name, '*.jpg'))
                frame_idx = [int(os.path.basename(x)[:-4]) for x in _frame_paths]
                frame_idx.sort()                
                _ref_masks = []
                _all_masks = []
                _ref_per_cls = {}
                k=2
                for obj in json_data['videos'][seq_name]['objects'].keys():
                    obj_masks = list(json_data['videos'][seq_name]['objects'][obj]['frames'])
                    _all_masks.extend(int(x) for x in obj_masks if int(x) not in _all_masks)
                    ref_idx = int(obj_masks[0])
                    _ref_per_cls[k] = ref_idx
                    k += 1 
                    if ref_idx not in _ref_masks:
                        _ref_masks.append(ref_idx)
                _ref_masks.sort()
                _all_masks.sort()
                self.ref_per_cls[seq_name] = _ref_per_cls
                self.all_masks[seq_name] = _all_masks
                self.ref_masks[seq_name] = _ref_masks
                self.frame_idx[seq_name] = [i for i in frame_idx if i >= _ref_masks[0]]
                self.num_frames[seq_name] = min(len(self.frame_idx[seq_name]), self.max_num_frames)                
                _mask = np.array(Image.open(os.path.join(
                    self.mask_dir, seq_name, '{:05d}.png'.format(_ref_masks[0]))).convert("P"))
                self.orig_shape[seq_name] = np.shape(_mask)
                _mask = to_resize(_mask, new_size=self.img_size)
                self.shape[seq_name] = np.shape(_mask)
                
                
                # if seq_name == '0062f687f1': #'03deb7ad95':
                #     print('orig_cls: ', self.orig_cls[seq_name], '\n')
                #     print('_ref_masks: ', _ref_masks)
                #     print('frame_idx: {} - {}'.format(self.frame_idx[seq_name][0],self.frame_idx[seq_name][-1]))
                #     print('num_frames: ', self.num_frames[seq_name])
                #     print('num_cls: ', self.num_cls[seq_name])
                #     print('all_masks: ', self.all_masks[seq_name])
    
    
    def __getitem__(self, index):
        seq_name = self.videos[index]
        ref_masks = self.ref_masks[seq_name]
        info = {}
        info['name'] = seq_name
        seq_len = self.num_frames[seq_name]
        info['num_frames'] = seq_len
        info['orig_cls'] = self.orig_cls[seq_name]
        info['orig_shape'] = self.orig_shape[seq_name]
        #num_cls = self.num_cls[seq_name]        
        h, w = self.shape[seq_name]
        f_idx = self.frame_idx[seq_name]
        info['ref_masks'] = []
        info['n_to_frame'] = {}
        
        img_seq = torch.zeros((seq_len, 3, h, w), dtype=torch.float32)
        spx_seq = torch.zeros((seq_len, 1, h, w), dtype=torch.int32)
        label_seq = torch.zeros((seq_len, 1, h, w), dtype=torch.int32)
        
        for n in range(seq_len):
            
            if f_idx[n] in self.all_masks[seq_name]:
                info['n_to_frame'][n] = f_idx[n]
            
            img_path = os.path.join(self.image_dir, seq_name, '{:05d}.jpg'.format(f_idx[n]))                        
            mask_path = os.path.join(self.mask_dir, seq_name, '{:05d}.png'.format(f_idx[0]))
            if f_idx[n] in ref_masks:
                mask_path = os.path.join(self.mask_dir, seq_name, '{:05d}.png'.format(f_idx[n]))
                info['ref_masks'].append(n)
            
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)/255.
            mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
            
            img = to_resize(img, new_size=self.img_size)
            mask = to_resize(mask, new_size=self.img_size)
            label = self.order_lbl_yt(mask, self.orig_cls[seq_name], ini_lbl=1).astype(np.int32)
            # first label (background) set to 1
            
            if self.spx_method == 'precomp':
                spx_path = os.path.join(self.spx_dir, seq_name, '{:05d}.png'.format(f_idx[n]))
                spx = np.array(Image.open(spx_path), dtype=np.int32)                            
            elif self.spx_method == 'slic':
                spx = slic(img, n_segments=self.slic_num, compactness=3, 
                            sigma=1, multichannel=True, convert2lab=True) + 1
            else:
                spx, _ = self.isec.segment(img)
            
            if self.save_spx:
                spx_path = os.path.join(self.spx_dir, seq_name)
                os.makedirs(spx_path, exist_ok=True)
                spx_path = os.path.join(spx_path, '{:05d}.png'.format(f_idx[n]))
                Image.fromarray(spx.astype(np.int32)).save(spx_path)
        
            #img = torch.from_numpy(img).permute(2,0,1)
            img_seq[n] = self.img_transf(img)
            spx_seq[n] = torch.from_numpy(spx).unsqueeze(0)
            label_seq[n] = torch.from_numpy(label).unsqueeze(0)
        
        num_cls = label_seq.max()
        
        return img_seq, spx_seq, label_seq, num_cls, info
    
    def order_lbl_yt(self, mask, orig_lbl, ini_lbl=1):    
        new_lbl = list(range(1,len(orig_lbl)+1))
        new_mask = np.zeros_like(mask)
        for k, c in enumerate(orig_lbl):
            new_mask[mask==c] = new_lbl[k]
        return new_mask + ini_lbl
            
    def __len__(self):
        return len(self.videos)

##################################################
def map_labels(mask, lbl_map=None):
    
    if not torch.is_tensor(mask): mask = torch.from_numpy(mask)
    if lbl_map is None: 
        lbl_map = {}        
        mask2 = reorder_labels(mask)
        num_cls = mask2.max().item()            
        for c in range(1, num_cls+1):
            tc = mask[mask2==c]
            if torch.count_nonzero(tc):
                lbl_map[c] = tc[0].item()
    else:
        mask2 = torch.ones_like(mask)
        for c, tc in lbl_map.items():
            mask2[mask==tc] = c
        num_cls = len(lbl_map)
        
    return mask2, num_cls, lbl_map    


def get_data_ddp(config, rank=None, world_size=None):
    if config.dataset.lower() not in ['msra10k', 'msra', 'davis']:
        raise RuntimeError('Unknown dataset: {}'.format(config.dataset))
    
    train_loader = None
    test_loader = None
    cfg_train = config
    cfg_test = cfg_train if config.config_test is None else config.config_test
    DAVIS_trainset = DAVIS_train_dataset #DAVIS_frame_dataset
    _testset = DAVIS_frame_dataset
    if 'video' in cfg_test.to_return:
        if cfg_train.test_dataset == 'davis':
            _testset = DAVIS_video_dataset
        elif cfg_train.test_dataset == 'youtube':
            _testset = YOUTUBE_video_dataset
    
    
    if cfg_train.dataset.lower() == 'davis':
        train_set = DAVIS_trainset(config, imset='train')
        test_set = _testset(config, imset='test')
    else:
        train_set = MSRA_dataset(cfg_train, imset='train')
        test_set = _testset(cfg_test, imset='test')
        
                    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, 
                                       drop_last=cfg_train.drop_last)
    train_loader = data.DataLoader(train_set, batch_size=cfg_train.train_batch_size, shuffle=False,
                                   num_workers=0, drop_last=cfg_train.drop_last, pin_memory=False, sampler=train_sampler)
    # test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False, 
    #                                    drop_last=False)#cfg_test.drop_last)
    
    test_sampler = DistributedEvalSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    
    test_loader = data.DataLoader(test_set, batch_size=cfg_test.test_batch_size, shuffle=False,
                                   num_workers=0, drop_last=False, pin_memory=False, sampler=test_sampler)        
    
        
    
    return train_loader, test_loader


if __name__ == "__main__":    
    
    print('\nYouTube-VOS corrupted data detection\n\n')
    
    from config_LT import set_config
    config = set_config()
    cfg_test = config.config_test

    yt_data = YOUTUBE_video_dataset(cfg_test, imset='valid')
    
    yt_loader = data.DataLoader(yt_data, batch_size=1, shuffle=False, num_workers=2)
    
    yt_len = len(yt_loader)
    
    print('yt_loader lenght: ', yt_len,'\n')
    
    dataiter = iter(yt_loader)
    
    for idx in range(yt_len):
        print('{}/{}', idx, yt_len)
        
        _, _, _, _, _ = dataiter.next()
        
    print('\n\nfinished!!!\n\n')




        
        
                
    
    
    
    
    
    
    
    
    
    
    


