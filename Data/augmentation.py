#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import math
import os.path
import numpy as np
from PIL import Image
import cv2
from skimage.measure import label as to_label
from isec_py.isec import shell_kernel, remove_borders
from scipy import stats
from Data.aug_davis import aug_heavy
import torch
import torchvision.transforms as transforms


class Data_Augmentation():
    def __init__(self, config, img_list, max_obj=4, split_data=None):
        
        self.img_list = img_list
        self.image_dir = config.msra_images
        self.mask_dir = config.msra_masks
        self.max_obj = max_obj
        self.img_size = config.train_img_size
        self.aug_affine = aug_heavy(self.img_size[0])
        self.split_data = split_data
        self.repeat_obj = 1/3
        
        
    def insert_random_objects(self, img, mask, num_obj=None, idx=None):
        
        mask, num_obj = remove_small_objcts(mask)
        
        if num_obj >= self.max_obj:
            return img, mask, num_obj
        
        if self.split_data and idx is not None:
            slc = int(len(self.img_list)/self.split_data)
            idx = idx + np.random.choice(list(range(0, slc*self.split_data, slc))+[len(self.img_list)-slc])
        elif self.split_data and idx is None:
            idx = int(len(self.img_list)/self.split_data)+1        
        elif not self.split_data and idx is None:        
                idx = len(self.img_list)+1
        
        new_img, new_mask = np.copy(img), np.copy(mask)
        h, w = mask.shape[0], mask.shape[1]
        
        min_obj=0 if num_obj>=2 else 2-num_obj
        num_insert = random.randint(min_obj, self.max_obj - num_obj)
        rand_img_list = random.sample(self.img_list[:idx]+self.img_list[idx+1:], num_insert)
        
        for rand_img in rand_img_list:
            
            img_path = os.path.join(self.image_dir, rand_img + ".jpg")        
            mask_path = os.path.join(self.mask_dir, rand_img + ".png") 
            
            img = np.array(Image.open(img_path).convert('RGB'), dtype="float32")/255.            
            mask = (np.array(Image.open(mask_path).convert('P')) > 0).astype(np.uint8)
            
            #img = cv2.resize(np.copy(img), (h,w))
            #mask = cv2.resize(np.copy(mask), (h,w), interpolation=cv2.INTER_NEAREST)
            
            obj_contour, _ = shell_kernel(mask)
            obj_label = remove_borders(to_label((1-obj_contour).astype(int), connectivity=1, return_num=False))
            obj_label, num_extra_obj = remove_small_objcts(obj_label)
            
            if num_extra_obj > 1:
                bb = bbox2(obj_label == random.randint(2,num_extra_obj))
                if bb is not None:
                    rmin, rmax, cmin, cmax = bb
                    obj_rgb = img[rmin : rmax, cmin : cmax]
                    obj_mask = mask[rmin : rmax, cmin : cmax]
                    
                    obj_rgb, obj_mask, _ = random_augmentations(obj_rgb, obj_mask, (h,w))
                    
                    #pad with 100 pixels (25 by side)
                    obj_rgb = img_pad(obj_rgb, 50)
                    obj_mask = img_pad(obj_mask, 50)  
                    
                
                    new_img, new_mask = random_paste(new_img, new_mask, obj_rgb, obj_mask*(new_mask.max()+1), 100)
                
        return new_img, new_mask, new_mask.max()
    
    def make_sequence(self, img, label, num_obj=None, idx=None, seq_len=3):
        
        label, num_obj = remove_small_objcts(label)
        
        img_seq = [cv2.resize(np.copy(img), self.img_size)]
        label_seq = [cv2.resize(np.copy(label), self.img_size, interpolation=cv2.INTER_NEAREST)]
        obj_numel = [np.bincount(label_seq[0].flatten())]
        for t in range(1,seq_len):
            new_img_, new_mask_ = self.aug_affine([jitter(np.copy(img))],[np.copy(label)])
            img_seq += new_img_
            label_seq += new_mask_
            obj_numel += [np.bincount(label_seq[t].flatten())]
        
        if num_obj >= self.max_obj:
            return img_seq, label_seq
        
        if idx is None:
            idx = len(self.img_list)+1
        
        min_obj=0 if num_obj>=2 else 2-num_obj
        num_insert = random.randint(min_obj, self.max_obj - num_obj)
        rand_img_list = random.sample(self.img_list[:idx]+self.img_list[idx+1:], num_insert)
        first_img = True
        
        for rand_img in rand_img_list:
            
            if first_img or random.random() > self.repeat_obj:
                img_path = os.path.join(self.image_dir, rand_img + ".jpg")        
                mask_path = os.path.join(self.mask_dir, rand_img + ".png")
                first_img = False
            
            img = np.array(Image.open(img_path).convert('RGB'), dtype="float32")/255.            
            #mask = (np.array(Image.open(mask_path).convert('P')) > 0).astype(np.uint8)
            mask = np.array(Image.open(mask_path).convert('P'))/255.
            
            obj_contour, _ = shell_kernel((mask>0).astype(np.uint8))
            obj_label = remove_borders(to_label((1-obj_contour).astype(int), connectivity=1, return_num=False))
            obj_label, num_extra_obj = remove_small_objcts(obj_label)
            
            if num_extra_obj > 1:
                bb = bbox2(obj_label == random.randint(2,num_extra_obj))
                if bb is not None:
                    rmin, rmax, cmin, cmax = bb
                    obj_rgb = img[rmin : rmax, cmin : cmax]
                    obj_mask = random_binarization(mask[rmin : rmax, cmin : cmax])
                    #obj_mask = (mask[rmin : rmax, cmin : cmax]>0).astype(np.uint8)
                    
                    new_obj_size = None
                    for t in range(seq_len):
                        # new_obj_rgb, new_obj_mask, new_obj_size = random_augmentations(jitter(np.copy(obj_rgb)), np.copy(obj_mask), 
                        #                                                                 self.img_size, new_obj_size)
                        
                        new_obj_rgb, new_obj_mask, new_obj_size = random_augmentations(obj_rgb, obj_mask, 
                                                                                        self.img_size, None)
                        obj_rgb, obj_mask = np.copy(jitter(new_obj_rgb)), np.copy(new_obj_mask)
                        
                        #pad with 100 pixels (25 by side)
                        hpad = min(2*int((rmax-rmin)/2), 50)
                        wpad = min(2*int((cmax-cmin)/2), 50)
                        new_obj_rgb = img_pad(new_obj_rgb, hpad, wpad)
                        new_obj_mask = img_pad(new_obj_mask, hpad, wpad)
                        
                        paste_area = None if t==0 else get_paste_area(label_seq[t-1], label_seq[t-1].max(), to_add=min(max(0.2, 0.2+t/10), 1))
                        
                        img_seq[t], label_seq[t], new_obj_numel = random_paste(img_seq[t], label_seq[t], new_obj_rgb,
                             new_obj_mask*(label_seq[t].max()+1), 2*hpad, 2*wpad, obj_numel[t], max_occlusion=0.5, paste_area=paste_area)
                        
                        if t==0 and (new_obj_numel.shape[0] == obj_numel[t].shape[0]): break
                        obj_numel[t] = np.append(obj_numel[t], new_obj_numel[-1])
                
        return img_seq, label_seq    


def random_binarization(mask):    
    #if random.random() < .25: return (mask>0).astype(np.uint8)     
    mask_h = mask > random.randint(5,10)/10
    mask_l = mask < random.randint(1,5)/10    
    new_mask = np.logical_and(mask_l, mask > 0)
    new_mask = np.logical_or(new_mask, mask_h)  
    if np.count_nonzero(new_mask) / np.count_nonzero(mask) < 0.1:
        return (mask>0).astype(np.uint8)
    return new_mask.astype(np.uint8)
    

def get_paste_area(mask, num_cls, to_add=0.2):
    rmin, rmax, cmin, cmax = bbox2(mask == num_cls)    
    #radd = int(((rmax-rmin) * to_add)/2)
    #cadd = int(((cmax-cmin) * to_add)/2)
    
    radd = int((mask.shape[0] * to_add)/2)
    cadd = int((mask.shape[1] * to_add)/2)
    
    nrmin = max(0, rmin-radd)
    nrmax = min(mask.shape[0], rmax+radd)
    ncmin = max(0, cmin-cadd)
    ncmax = min(mask.shape[1], cmax+cadd)       
    return nrmin, nrmax, ncmin, ncmax   
    

def remove_small_objcts(mask, num_obj=None, min_size=0.001):
    
    if num_obj is None:
        num_obj = np.max(mask)
    
    if num_obj < 2:
        return mask, num_obj
    
    h, w = mask.shape[0], mask.shape[1]
    new_mask = np.copy(mask)
    
    skips = 0
    for k in range(1, num_obj+1):
        
        obj_mask = (mask == k).astype(np.uint8)
        
        if np.count_nonzero(obj_mask)/(h*w) < min_size:
            bb = bbox2(obj_mask)
            if bb is not None:        
                rmin, rmax, cmin, cmax = bb
                rmin, rmax, cmin, cmax = max(0,rmin-1), min(h,rmax+1), max(0,cmin-1), min(w,cmax+1)
                obj_mask = mask[rmin : rmax, cmin : cmax] 
                x = obj_mask[obj_mask != k]
                m = stats.mode(x, axis=None)
                new_mask = np.where(mask == k, m[0][0], new_mask)
            else:
                new_mask = np.where(mask == k, 1, new_mask)
            
            num_obj -= 1            
            if skips > 0:
                new_mask = np.where(new_mask >= k, new_mask-skips, new_mask)
                skips = 0
        else:
            skips += 1
    
    return new_mask, num_obj
            

def random_augmentations(img, mask, img_size, obj_size=None):
    
    # Random rescaling
    rf = (0.1, 0.6)
    obj_h, obj_w = mask.shape    
    ref_h, ref_w = img_size
    min_h, min_w = int(rf[0]*ref_h), int(rf[0]*ref_w)
    max_h, max_w = int(rf[1]*ref_h), int(rf[1]*ref_w)         
    if obj_size is not None:
        rf = (0.85, 1.15)
        ref_h, ref_w = obj_size
            
    if obj_h <= obj_w:
        new_h = random.randint(math.ceil(rf[0]*ref_h), math.ceil(rf[1]*ref_h))
        new_h = max(min_h, min(new_h, max_h))
        new_w = int((new_h/obj_h)*obj_w)
    else:
        new_w = random.randint(math.ceil(rf[0]*ref_w), math.ceil(rf[1]*ref_w))
        new_w = max(min_w, min(new_w, max_w))
        new_h = int((new_w/obj_w)*obj_h)        
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    obj_size = (new_h, new_w)
            

    # Random flipping
    # if random.random() < 0.0:
    #     img = np.flip(img, axis=0).copy()
    #     mask = np.flip(mask, axis=0).copy()
    # if random.random() < 0.0:
    #     img = np.flip(img, axis=1).copy()  
    #     mask = np.flip(mask, axis=1).copy()
        
    # Affine
    if True: # random.random() < 0.8:
        h, w = mask.shape[0], mask.shape[1]        
        dst_points = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
        tx1 = random.randint(-w//10,w//10)
        ty1 = random.randint(-h//10,h//10)
        tx2 = random.randint(-w//10,w//10)
        ty2 = random.randint(-h//10,h//10)
        tx3 = random.randint(-w//10,w//10)
        ty3 = random.randint(-h//10,h//10)
        tx4 = random.randint(-w//10,w//10)
        ty4 = random.randint(-h//10,h//10)
        src_points = np.float32([[0 + tx1,0 + ty1],[0 + tx2,h-1 + ty2],
                                 [w-1 + tx3,h-1 + ty3],[w-1 + tx4,0 + ty4]])
        H1,_ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10)        
        img = cv2.warpPerspective(img, H1, (w,h),flags = cv2.INTER_LINEAR) 
        mask = cv2.warpPerspective(mask, H1, (w,h),flags = cv2.INTER_NEAREST)  
    
    return img, mask, obj_size

def jitter(img, b=0.2, c=0.2, s=0.2, h=0.01):
    """
    Randomly alter brightness, contrast, saturation, hue within given range
    """    
    img = torch.from_numpy(img)    
    transform = transforms.ColorJitter(
    brightness=b, contrast=c, saturation=s, hue=h)  
    img = transform(img.permute(2,0,1))        
    return img.permute(1,2,0).numpy()
            
            
def bbox2(img):
    try:
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    
        return rmin, rmax+1, cmin, cmax+1
    except:
        return None
        
        
def img_unpad(img, hpad, wpad=None):    
    
    if wpad is None:
        wpad = hpad
    
    h, w = img.shape[0], img.shape[1]
    h_dif = np.floor((hpad)/2).astype(np.uint8)
    w_dif = np.floor((wpad)/2).astype(np.uint8)
         
    
    return img[h_dif : h-h_dif, w_dif : w-w_dif]

def img_pad(img, hpad, wpad=None):    
    
    if wpad is None:
        wpad = hpad
    
    to_pad = (img.shape[0]+hpad, img.shape[1]+wpad)
    
    if len(img.shape) == 3:
        to_pad = to_pad+(0,)    
    
    shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(img.shape,to_pad)]
    shape_diffs = np.maximum(shape_diffs,0)
    pad_sizes = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
    
    return np.pad(img, pad_sizes, mode='constant')
        

def random_paste(frame_canvas, mask_canvas, frame, mask, hpad, wpad=None, obj_numel=None, max_occlusion=None, paste_area=None):
    
    if wpad is None: wpad = hpad
    
    if obj_numel is None and max_occlusion is not None: 
        obj_numel = np.bincount(mask_canvas.flatten())
        
    num_attempts = 3
    for n in range(num_attempts):
        
        new_frame = np.copy(frame_canvas)
        new_mask = np.copy(mask_canvas)
        
        h_canvas, w_canvas = new_frame.shape[0], new_frame.shape[1]
        h_paste, w_paste = frame.shape[0], frame.shape[1]
        
        if paste_area is not None:
            rmin, rmax, cmin, cmax = paste_area
        else:
            rmin, rmax, cmin, cmax = 0, h_canvas, 0, w_canvas      
        
        h_canvas_pos = random.randint(rmin, max(rmin, rmax - h_paste + hpad))
        w_canvas_pos = random.randint(cmin, max(cmin, cmax - w_paste + wpad))
        
        h_canvas_ini = max(0, h_canvas_pos - hpad//2)
        w_canvas_ini = max(0, w_canvas_pos - wpad//2)
        
        h_canvas_fin = min(h_canvas, h_canvas_pos + h_paste - hpad//2)
        w_canvas_fin = min(w_canvas, w_canvas_pos + w_paste - wpad//2)        
        
        h_paste_pos_R = h_canvas_fin - h_canvas_pos
        w_paste_pos_R = w_canvas_fin - w_canvas_pos        
        
        h_paste_pos_L = h_canvas_pos - h_canvas_ini
        w_paste_pos_L = w_canvas_pos - w_canvas_ini
        
        h_paste_ini = hpad//2 - h_paste_pos_L
        w_paste_ini = wpad//2 - w_paste_pos_L
        
        h_paste_fin = hpad//2 + h_paste_pos_R
        w_paste_fin = wpad//2 + w_paste_pos_R        
        
        hci, hcf, wci, wcf = h_canvas_ini, h_canvas_fin, w_canvas_ini, w_canvas_fin
        hpi, hpf, wpi, wpf = h_paste_ini, h_paste_fin, w_paste_ini, w_paste_fin
        
        new_frame[hci:hcf, wci:wcf] = new_frame[hci:hcf, wci:wcf] * ~(mask[hpi:hpf, wpi:wpf, None] > 0)
        new_frame[hci:hcf, wci:wcf] = new_frame[hci:hcf, wci:wcf] + frame[hpi:hpf, wpi:wpf] * (mask[hpi:hpf, wpi:wpf, None] > 0)
        
        new_mask[hci:hcf, wci:wcf] = new_mask[hci:hcf, wci:wcf] * ~(mask[hpi:hpf, wpi:wpf] > 0)                    
        new_mask[hci:hcf, wci:wcf] = new_mask[hci:hcf, wci:wcf] + mask[hpi:hpf, wpi:wpf]
        
        if max_occlusion is None: return new_frame, new_mask
        
        new_obj_numel = np.bincount(new_mask.flatten())        
        np.seterr(invalid='ignore')
        if new_obj_numel[2:-1].shape == obj_numel[2:].shape:            
            if ((new_obj_numel[2:-1]/obj_numel[2:]) > max_occlusion).all():
                return new_frame, new_mask, np.append(obj_numel, new_obj_numel[-1])
       
    return frame_canvas, mask_canvas, obj_numel




