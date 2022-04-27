#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import math
import os.path
import numpy as np
from PIL import Image
import cv2
from skimage.measure import label
from isec_py.isec import shell_kernel, remove_borders
from scipy import stats


class Data_Augmentation():
    def __init__(self, config, img_list, max_obj=4):
        
        self.img_list = img_list
        self.image_dir = config.msra_images
        self.mask_dir = config.msra_masks
        self.max_obj = max_obj
        
        
    def insert_random_objects(self, img, mask, num_obj=None, idx=None):
        
        mask, num_obj = remove_small_objcts(mask)
        
        if num_obj >= self.max_obj:
            return img, mask, num_obj
        
        if idx is None:
            idx = len(self.img_list)+1
        
        new_img, new_mask = np.copy(img), np.copy(mask)
        w, h = mask.shape[0], mask.shape[1]
        
        num_insert = random.randint(0, self.max_obj - num_obj)
        rand_img_list = random.sample(self.img_list[:idx]+self.img_list[idx+1:], num_insert)
        
        for rand_img in rand_img_list:
            
            img_path = os.path.join(self.image_dir, rand_img + ".jpg")        
            mask_path = os.path.join(self.mask_dir, rand_img + ".png") 
            
            img = np.array(Image.open(img_path).convert('RGB'), dtype="float32")/255.            
            mask = (np.array(Image.open(mask_path).convert('P')) > 0).astype(np.uint8)
            
            #img = cv2.resize(np.copy(img), (w,h))
            #mask = cv2.resize(np.copy(mask), (w,h), interpolation=cv2.INTER_NEAREST)
            
            obj_contour, _ = shell_kernel(mask)
            obj_label = remove_borders(label((1-obj_contour).astype(int), connectivity=1, return_num=False))
            obj_label, num_extra_obj = remove_small_objcts(obj_label)
            
            if num_extra_obj > 1:
                bb = bbox2(obj_label == random.randint(2,num_extra_obj))
                if bb is not None:
                    rmin, rmax, cmin, cmax = bb
                    obj_rgb = img[rmin : rmax, cmin : cmax]
                    obj_mask = mask[rmin : rmax, cmin : cmax]
                    
                    obj_rgb, obj_mask = random_augmentations(obj_rgb, obj_mask, (w,h))
                    
                    #pad with 100 pixels (25 by side)
                    obj_rgb = img_pad(obj_rgb, 50)
                    obj_mask = img_pad(obj_mask, 50)  
                    
                
                    new_img, new_mask = random_paste(new_img, new_mask, obj_rgb, obj_mask*(new_mask.max()+1), 100)
                
        return new_img, new_mask, new_mask.max()
            

def remove_small_objcts(mask, num_obj=None, min_size=0.01):
    
    if num_obj is None:
        num_obj = np.max(mask)
    
    if num_obj < 2:
        return mask, num_obj
    
    w, h = mask.shape[0], mask.shape[1]
    new_mask = np.copy(mask)
    
    skips = 0
    for k in range(1, num_obj+1):
        
        obj_mask = (mask == k).astype(np.uint8)
        
        if np.count_nonzero(obj_mask)/(mask.shape[0]*mask.shape[1]) < min_size:
            bb = bbox2(obj_mask)
            if bb is not None:        
                rmin, rmax, cmin, cmax = bb
                rmin, rmax, cmin, cmax = max(0,rmin-1), min(w,rmax+1), max(0,cmin-1), min(h,cmax+1)
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
            

def random_augmentations(img, mask, shape):
    
    w, h = shape    

    # Random rescaling       
    if w <= h:
        randon_size = random.randint(math.ceil(0.2*w), math.ceil(0.7*w))
        factor = randon_size/w
        img = cv2.resize(img, (randon_size, int(factor * h)), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (randon_size, int(factor * h)), interpolation=cv2.INTER_NEAREST)             
    else:
        randon_size = random.randint(math.ceil(0.2*h), math.ceil(0.7*h))
        factor = randon_size/h
        img = cv2.resize(img, (int(factor * w), randon_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (int(factor * w), randon_size), interpolation=cv2.INTER_NEAREST)           

    # Random flipping
    if random.random() < 0.5:
        img = np.flip(img, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()
    if random.random() < 0.5:
        img = np.flip(img, axis=1).copy()  
        mask = np.flip(mask, axis=1).copy()
        
    # Affine
    if random.random() < 0.5:
        w, h = mask.shape[0], mask.shape[1]
        
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
    
    return img, mask
            
            
def bbox2(img):
    try:
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    
        return rmin, rmax+1, cmin, cmax+1
    except:
        return None
        
        
def img_unpad(img, wpad, hpad=None):    
    
    if hpad is None:
        hpad = wpad
    
    w, h = img.shape[0], img.shape[1]    
    w_dif = np.floor((wpad)/2).astype(np.uint8)
    h_dif = np.floor((hpad)/2).astype(np.uint8)     
    
    return img[w_dif : w-w_dif, h_dif : h-h_dif]

def img_pad(img, wpad, hpad=None):    
    
    if hpad is None:
        hpad = wpad
    
    to_pad = (img.shape[0]+wpad, img.shape[1]+hpad)
    
    if len(img.shape) == 3:
        to_pad = to_pad+(0,)    
    
    shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(img.shape,to_pad)]
    shape_diffs = np.maximum(shape_diffs,0)
    pad_sizes = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
    
    return np.pad(img, pad_sizes, mode='constant')
        

def random_paste(frame_canvas, mask_canvas, frame, mask, wpad, hpad=None):
    
    if hpad is None:
        hpad = wpad
        
    w_canvas, h_canvas = frame_canvas.shape[0], frame_canvas.shape[1]
    w_paste, h_paste = frame.shape[0], frame.shape[1]
    
    w_canvas_pos = random.randint(0, max(0, w_canvas - w_paste + wpad))
    h_canvas_pos = random.randint(0, max(0, h_canvas - h_paste + hpad))
    
    w_canvas_ini = max(0, w_canvas_pos - wpad//2)
    h_canvas_ini = max(0, h_canvas_pos - hpad//2)
    
    w_canvas_fin = min(w_canvas, w_canvas_pos + w_paste - wpad//2)
    h_canvas_fin = min(h_canvas, h_canvas_pos + h_paste - hpad//2)
    
    w_paste_pos_R = w_canvas_fin - w_canvas_pos
    h_paste_pos_R = h_canvas_fin - h_canvas_pos
    
    w_paste_pos_L = w_canvas_pos - w_canvas_ini
    h_paste_pos_L = h_canvas_pos - h_canvas_ini
    
    w_paste_ini = wpad//2 - w_paste_pos_L
    h_paste_ini = hpad//2 - h_paste_pos_L
    
    w_paste_fin = wpad//2 + w_paste_pos_R
    h_paste_fin = hpad//2 + h_paste_pos_R
    
    wci, wcf, hci, hcf = w_canvas_ini, w_canvas_fin, h_canvas_ini, h_canvas_fin
    wpi, wpf, hpi, hpf = w_paste_ini, w_paste_fin, h_paste_ini, h_paste_fin
    
    frame_canvas[wci:wcf, hci:hcf] = frame_canvas[wci:wcf, hci:hcf] * ~(mask[wpi:wpf, hpi:hpf, None] > 0)    
    frame_canvas[wci:wcf, hci:hcf] = frame_canvas[wci:wcf, hci:hcf] + frame[wpi:wpf, hpi:hpf] * (mask[wpi:wpf, hpi:hpf, None] > 0)
    
    mask_canvas[wci:wcf, hci:hcf] = mask_canvas[wci:wcf, hci:hcf] * ~(mask[wpi:wpf, hpi:hpf] > 0)    
    mask_canvas[wci:wcf, hci:hcf] = mask_canvas[wci:wcf, hci:hcf] + mask[wpi:wpf, hpi:hpf]    
    
    return frame_canvas, mask_canvas


