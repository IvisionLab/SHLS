#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import time
import numpy as np
import cv2
import random
from skimage.measure import label as to_label
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import contextmanager
from math import ceil, floor
from sklearn.model_selection import train_test_split
import faiss
import faiss.contrib.torch_utils
from sklearn.utils.extmath import weighted_mode
from skimage.morphology import convex_hull_image

#import multiprocessing
#faiss.omp_set_num_threads(multiprocessing.cpu_count())
faiss.omp_set_num_threads(1)

device = torch.device("cpu")


@contextmanager
def context(condition, context_manager):
    if condition:
        with context_manager:
            yield
    else:
        yield

def set_seed(seed=5):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def launch_cuda_ddp(model, rank, optimizer=None, broadcast_buffers=True):
    global device
    device=torch.device("cuda:{}".format(rank))
    if rank == 0:
        print('CUDA {} [Pytorch DDP using {} device(s)]'.format(torch.version.cuda, torch.cuda.device_count()))
    model = model.to(rank)    
    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=False, broadcast_buffers=broadcast_buffers)
    
    if optimizer is not None:
        optimizer = optimizer_ddp(optimizer,device) 
        return model, optimizer
    
    return model


def optimizer_ddp(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optim


def load_checkpoint(config, model, optimizer=None, rank=None, train=False):
    # load saved model if specified
    load_name = os.path.join(config.resume_model_path)    
    if not os.path.isfile(load_name):
        raise RuntimeError('File not found: {}'.format(load_name))       
    
    if rank is not None:
        checkpoint = torch.load(load_name, map_location=torch.device("cuda:{}".format(rank)))
    else:
        checkpoint = torch.load(load_name) # dict_keys(['epoch', 'model', 'optimizer'])
    
    start_epoch = checkpoint['epoch'] + 1
    try:        
        top_iou = checkpoint['top_iou']
        global_ite = checkpoint['global_ite']
        test_ite = checkpoint['test_ite']
    except:
        top_iou, global_ite, test_ite = 0, 0, 0
    
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model']
    pretrained_dict_update = {}
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
    
    model_dict.update(pretrained_dict_update)
    model.load_state_dict(model_dict, strict=False)    

    if rank is None or rank==0:
        print('Loaded checkpoing at: {}'.format(load_name))
    
    if config.reset_iou:
        top_iou = 0.0
    
    if optimizer is not None:
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        del(checkpoint)
        if train:
            return model, optimizer, start_epoch, top_iou, global_ite, test_ite
        else:
            return model, optimizer
    
    del(checkpoint)
    if train:
        return model, start_epoch, top_iou, global_ite, test_ite
    else:
        return model

def save_model(config, model, model_name, epoch=0, optimizer=None, iou=0.0, top_iou=0.0,
               global_ite=0, test_ite=0, loss=0.0, elapsed_time=None, log=''):
    
    #if not os.path.exists(config.save_model_path):
    os.makedirs(config.save_model_path, exist_ok=True)
    
    path_ = os.path.join(config.save_model_path, config.model + model_name)
    
    if optimizer is not None:    
        torch.save({'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'top_iou': top_iou,
                    'global_ite': global_ite,
                    'test_ite': test_ite,}, path_)
    else:
        torch.save({'model': model.state_dict(),
                    'epoch': epoch,
                    'top_iou': top_iou,
                    'global_ite': global_ite,
                    'test_ite': test_ite,}, path_)
    print('Model saved in {}'.format(path_))
    
     # Append log
    if model_name == '_last.pth':
        log_path = os.path.join(config.save_model_path, 
                                config.model + '_' + config.current_time + '.log')    
        with open(log_path, 'a') as f:
            if epoch == 0:
                f.write('\n\n[{}]:'.format(config.experiment_name))
                f.write('\n______________________________________________')                
            if iou == top_iou:
                log += ' [TOP]'
            f.write('\n'+log)
            #     f.write('\n[Epoch:{}]: avg_loss: {:^7.3f}, J&F: {:^7.3f} [TOP]'.format(epoch+1, loss, iou))
            # else:
            #     f.write('\n[Epoch:{}]: avg_loss: {:^7.3f}, J&F: {:^7.3f}'.format(epoch+1, loss, iou))
            if elapsed_time: f.write('\nElapsed time: '+ elapsed_time)


def freeze_batchnorm(model, freeze_modules):    
    if not freeze_modules: return
    
    M = model
    if isinstance(model, DDP): M = model.module   
    
    for fm in freeze_modules:        
        for (name, module) in M.named_children():
            if name.find(fm) != -1:
                for md in module.modules():
                    if isinstance(md, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        if hasattr(md, 'weight'):
                            md.weight.requires_grad_(False)
                        if hasattr(md, 'bias'):
                            md.bias.requires_grad_(False)
                        md.eval()
                        #md.track_running_stats = False
                        

def spx_info_map(labels):
    # info_map = [b,4,w,h] -> ch0=spx_size, ch1=spx_xmap, ch2=spx_ymap, ch3=spx_idx_normalized
    labels = labels.int()
    b, c, h, w = labels.shape
    info_map = torch.zeros([b,3,h,w], device=device)    
    xvalues = torch.arange(0,h, device=device)
    yvalues = torch.arange(0,w, device=device)
    xx, yy = torch.meshgrid(xvalues/(h-1), yvalues/(w-1))    
    for b_idx in range(b):
        label_max = labels[b_idx].max()
        for n in range(1, label_max+1):
            spx_size = torch.nonzero(labels[b_idx]==n).shape[0] / (h*w)
            info_map[b_idx,0,labels[b_idx,0]==n] = spx_size            
            info_map[b_idx,1,labels[b_idx,0]==n] = xx[labels[b_idx,0]==n].mean()
            info_map[b_idx,2,labels[b_idx,0]==n] = yy[labels[b_idx,0]==n].mean()
        #info_map[b_idx,3] = labels[b_idx] / label_max
    return info_map


def get_confident_pools(spx, seg, pred, num_cls, r=0.4):
    batch_size, _, _, _ = spx.shape
    conf_pools = []    
    for b in range(batch_size):
        conf = False
        num_spx = spx[b,0].max().item()
        val = torch.zeros(num_spx, dtype=spx.dtype, device=device)
        for c in range(num_cls):
            if c == 0:
                pred_pad = pred[b,c,seg[b,0]==c+1]
                try:
                    pad = pred_pad.max().item()
                except:
                    return None
                m_cls = pred_pad[pred_pad < pad].mean()
            else:                
                m_cls = pred[b,c,seg[b,0]==c+1].mean()            
            for s in range(num_spx):                
                m_spx = pred[b,c,spx[b,0]==s+1].mean()                
                if m_spx >= m_cls * r:                    
                    val[s] = 1 
                    conf = True
        conf_pools.append(val)
        if not conf:
            return None
    return conf_pools


def get_spx_pools(spx, label, merge=False):
# output: spx per obj (greater intersection = statistic mode)
# merge=True make spx adhere to labels borders
# [spx, label]    
    batch_size, _, _, _ = spx.shape
    pools = []
    if merge: new_spx = torch.zeros_like(spx)
    
    for b in range(batch_size):
        spx_ = spx.clone()
        to_merge = merge        
        for _ in range(2):            
            num_spx = spx_[b,0].max().item()           
            p = torch.zeros(num_spx, dtype=spx.dtype, device=device)            
            for s in range(num_spx):    
                p[s] = label[b,0,spx_[b,0]==s+1].mode()[0]    
                if to_merge: new_spx[b,0,spx_[b,0]==s+1] = p[s]                
            if to_merge:
                new_spx[b] = merge_spx_label(spx_[b], label[b], new_spx[b])
                #new_spx[b] = new_merge(spx_[b], label[b], p)
                spx_[b] = new_spx[b].clone()
                to_merge = False
                continue                        
            pools.append(p)
            if not merge: break
        
    if merge: return pools, new_spx
    return pools


def merge_spx_label(spx, label, spx2label=None):
    
    if spx2label is None:
        spx2label = label.clone()
    
    num_spx = spx.max().item()
    output = torch.zeros_like(spx, device=device)
    
    for i in range(1, label.max()+1):        
        n_spx = spx.clone()
        n_spx[spx2label != i] = 0
        
        n_obj_label = label.clone() + num_spx
        n_obj_label[label != i] = 0
        
        r_spx = n_spx + n_obj_label        
        x_spx = r_spx.clone()        
        x_spx[r_spx != num_spx+i] = 0
        r_spx = r_spx - num_spx - i
        r_spx[label != i] = 0
        r_spx[x_spx == num_spx+i] = 0        
        z_spx = torch.from_numpy(to_label((x_spx>0).cpu().numpy().astype(np.int32), 
                                       connectivity=1, return_num=False)).to(device)        
        r_spx = r_spx + x_spx + z_spx
        
        output = output + r_spx
        
        num_spx = output.max().item()
    
    return reorder_labels(output)


def reorder_labels(labels, first_label=1):
    # reorder the sequence of labels by removing inexistent values
    all_labels = torch.tensor(range(first_label, labels.max()+1))
    present_labels = all_labels.clone().apply_(lambda x: x in labels)    
    gap = 0
    cor = 0
    for i, k in enumerate(all_labels):            
        if present_labels[i] == 0:
            gap += 1
            cor += 1
        else:
            labels[labels>=k-cor] -= gap
            gap = 0    
    return labels


def to_onehot(mask, num_cls, ini_cls=1):
    # input: mask [B,1,w,h]
    # output: onehot [B,num_cls,w,h]    
    B, _, h, w = mask.shape
    onehot = torch.zeros((B,num_cls,h,w), dtype=mask.dtype, device=mask.device)    
    for b in range(B):
        for k in range(ini_cls, ini_cls+num_cls):
            onehot[b,k-ini_cls] = (mask[b,0] == k).to(mask.dtype)
    return onehot

def to_resize(img, new_size=(384,384)):   
    # new_size: tuple for fixed size resizing, int for keeping aspect ratio
    if new_size is None:
        return img
    
    img_h, img_w = img.shape[0], img.shape[1]
    
    if isinstance(new_size, tuple):
        hsize, wsize = new_size
    else:          
        if img_w >= img_h:
            wsize = new_size
            hsize = int((float(img_h)*float(wsize/float(img_w))))        
        else:
            hsize = new_size
            wsize = int((float(img_w)*float(hsize/float(img_h))))
        
    if len(img.shape) > 2 and img.shape[2] > 1:
        return cv2.resize(img, (wsize,hsize))
    else:
        return cv2.resize(img, (wsize,hsize), interpolation=cv2.INTER_NEAREST)


def location_map2(H, W, coords):
    ys, xs = np.ogrid[0:1:H*1j, 0:1:W*1j]    
    loc_map = np.zeros((H, W))
    for coord in coords:
        dist = 1 - np.hypot(xs-coord[1]/W, ys-coord[0]/H)/1.4142135623730951
        loc_map = loc_map + dist
    return torch.from_numpy(loc_map/len(coords))

def location_map(H, W, bbox, f=None):    
    rmin, rmax, cmin, cmax = bbox
    coords = [(rmin+int((rmax-rmin)/2), cmin+int((cmax-cmin)/2))]    
    if f is None:
        area = ((rmax-rmin)*(cmax-cmin))/(H*W) 
        f = max(1, min(int((1-area)*7), 5))    
    ys, xs = np.ogrid[0:1:H*1j, 0:1:W*1j]    
    loc_map = np.zeros((H, W))
    for coord in coords:
        dist = (1 - np.hypot(xs-coord[1]/W, ys-coord[0]/H)/1.4142135623730951)**f
        loc_map = loc_map + dist
    return torch.from_numpy(loc_map/len(coords))


def filt_outliers(seg, lastseg, num_cls, u=0.5):   
    seg_tensor = torch.is_tensor(seg)
    seg_device = seg.device
    if seg_tensor:
        seg = seg.cpu().numpy()
        lastseg = lastseg.cpu().numpy()
    n_seg = seg.copy()         
    for c in range (2, num_cls+1):            
        c_seg = (seg==c).astype(np.int32)            
        c_lbl, L = to_label(c_seg, connectivity=1, return_num=True)            
        cs = (lastseg==c).astype(np.int32).ravel()
        s1 = max(1, np.count_nonzero(cs))            
        s2 = np.bincount(c_lbl.ravel() * cs)
        s2[0] = 0
        a = s2/s1
        vs = np.nonzero(a > u)[0]            
        for x in vs:
            c_lbl[c_lbl==x] = -1            
        res = convex_hull_image((c_lbl == -1).astype(np.int32))            
        to_filt = np.logical_and(seg==c, res==0)            
        n_seg[to_filt] = 0    
    if seg_tensor: return torch.from_numpy(n_seg).to(seg_device)    
    return n_seg    


class FaissKNeighbors():
    def __init__(self, k=5, r=0.5):
        self.index = None
        self.y = None
        self.k = k
        self.num_cls = None
        self.dist_func = faiss.IndexFlatIP # Inner product for cosine similarity
        self.w = torch.tensor([r**x for x in range(self.k)])
        self.w_sum = self.w.sum()

    def fit(self, X, y, num_cls=None):
        self.y = y.cpu()
        self.num_cls = y.max().item() if num_cls is None else num_cls
        self.index = self.dist_func(X.shape[1])
        #self.index.reset()
        self.index.add(torch.nn.functional.normalize(X.float().cpu(), p=2, dim=1))
        
    def predict(self, X, dist_neigh=None):        
        k = self.k #if dist_neigh is None else dist_neigh        
        #distances, indices = self.index.search(faiss.normalize_L2(X.float().cpu()), k=self.k)
        distances, indices = self.index.search(torch.nn.functional.normalize(X.float().cpu(), p=2, dim=1), k=k)
        
        #indices = indices.to(device)
        votes = self.y[indices[:, :self.k]]
        #predictions = torch.mode(votes).values.int().to(X.device)
        w_mode = weighted_mode(votes.numpy(), self.w, axis=1)[0]
        predictions = torch.from_numpy(w_mode[:,0]).int().to(X.device)
        
        if dist_neigh is not None:
            return predictions, self.dist_per_class2(distances, indices).to(device=X.device)   
        return predictions

    def score(self, X, y):        
        score = torch.mean((y == self.predict(X)).float())
        return score
    
    def dist_per_class(self, dist, indi):        
        dist_cls = torch.zeros([dist.shape[0], self.num_cls], dtype=torch.float32)        
        for c in range(self.num_cls):            
            yc = (self.y[indi] == c+1).int()
            dist_cls[:,c] = torch.mean(torch.sort(dist*yc, dim=1, descending=True).values[:,:10], dim=1)            
        return dist_cls
    
    def dist_per_class2(self, dist, indi):        
        dist_cls = torch.zeros([dist.shape[0], self.num_cls], dtype=torch.float32)        
        for c in range(self.num_cls):            
            yc = (self.y[indi] == c+1).int()
            v = torch.sort(dist*yc, dim=1, descending=True).values
            wv = torch.mul(self.w, v)            
            dist_cls[:,c] = torch.sum(wv, dim=1)/self.w_sum            
        return dist_cls
            

class FaissKMeans():
    def __init__(self, n_clusters=10, n_init=10, max_iter=300, seed=1234):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.centroids = None        
        self.spherical = True
        self.tensor = False
        self.pca_dim = None
        self.pca_mat = None
        self.seed = seed
        self.device = None

    def fit(self, X, init_centroids=None):            
        self.tensor = torch.is_tensor(X)
        self.device = X.device
        nX = X.float().cpu().numpy() if self.tensor else X.copy().astype(np.float32)
        faiss.normalize_L2(nX)        
        if self.pca_dim: nX = self.pca(nX)        
        self.kmeans = faiss.Kmeans(d=nX.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   min_points_per_centroid=1,
                                   spherical=self.spherical,
                                   seed=self.seed)
        nX = self.check_pad(nX)
        self.kmeans.train(nX, init_centroids=init_centroids)
        self.centroids = self.kmeans.centroids        

    def predict(self, X):
        nX = X.float().cpu().numpy() if torch.is_tensor(X) else X.copy().astype(np.float32)
        faiss.normalize_L2(nX)
        if self.pca_dim: nX = self.pca(nX)
        if self.tensor: return torch.tensor(self.kmeans.index.search(nX, 1)[0]).to(self.device)
        return self.kmeans.index.search(nX, 1)[0]
    
    def pairwise_dist(self, X, clusters=None):
        nX = X.float().cpu().numpy() if torch.is_tensor(X) else X.copy().astype(np.float32)
        #faiss.normalize_L2(nX)
        if self.pca_dim: nX = self.pca(nX)
        if clusters is None: clusters = self.centroids
        cd = faiss.pairwise_distances(nX, clusters)        
        if self.tensor: return torch.from_numpy(cd).to(self.device)#.max(dim=1).values
        return cd
        
    def pca(self, nX):
        if self.pca_mat is None:
            self.pca_mat = faiss.PCAMatrix(nX.shape[1], self.pca_dim, 0, True)
            self.pca_mat.train(nX)
        return self.pca_mat.apply_py(nX)
    
    def check_pad(self, nX):
        pad = self.n_clusters - nX.shape[0]        
        if pad <= 0:
            return nX
        else:
            return np.pad(nX, ((0,pad),(0,0)), 'mean')


def tensor_train_test_split(x, y, test_size=0.8):
    if torch.is_tensor(x):
        x_np = x.cpu().numpy()
    if torch.is_tensor(y):
        y_np = y.cpu().numpy()    
    idx_np = np.arange(1, y_np.shape[0]+1)
    
    x_train, x_test, y_train, y_test, idx_train, idx_test = \
        train_test_split(x_np, y_np, idx_np, test_size=test_size, 
                         stratify=try_stratify(y_np, test_size))        
                        
    x_train = torch.from_numpy(x_train).to(x.device)
    x_test = torch.from_numpy(x_test).to(x.device)
    y_train = torch.from_numpy(y_train).to(y.device)
    y_test = torch.from_numpy(y_test).to(y.device)
    idx_train = torch.from_numpy(idx_train).to(x.device)
    idx_test = torch.from_numpy(idx_test).to(x.device)    
    
    return x_train, x_test, y_train, y_test, idx_train, idx_test

def try_stratify(y, test_size):
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_counts = np.bincount(y_indices)
    min_class = np.min(class_counts)
    n_samples = y.shape[0]
    train_size=1-test_size
    n_test = ceil(test_size * n_samples)
    n_train = floor(train_size * n_samples)
    n_train, n_test = int(n_train), int(n_test)
    
    if (n_train<n_classes) or (n_test<n_classes) or (min_class<2):
        return None
    else:
        return y
    
def show_intro(config, delay=0.1, size=20):
    print('\n[{}: {}] >\b'.format(config.model, config.experiment_name), end="", flush=True)
    for i in range(size):
        print('= >\b', end="", flush=True)
        time.sleep(delay)
    print('> [started]')

def format_time(sec):
    hours = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    ds = '0' if seconds<10 else ''
    dm = '0' if minutes<10 else ''
    dh = '0' if hours<10 else ''    
    return "%s%d:%s%d:%s%d" % (dh, hours, dm, minutes,ds, seconds)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#plt.ion()
def plot_grad_flow(named_parameters):    
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                a = p.grad.abs().mean()
            else:
                a = 0.0
            ave_grads.append(a)
            print('{} = {}'.format(n,a))
    # import matplotlib.pyplot as plt
    # plt.figure(num=1) 
    # plt.plot(ave_grads, alpha=0.8, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    # plt.draw()
    # plt.savefig('grad_plot_dec2.png')  
    # #plt.show() 
    # plt.close()

def print_peak_memory(prefix, device):
    if device==0:
        print('{}:{}MB'.format(
            prefix, (torch.cuda.max_memory_allocated(device)//1e6)))

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))
    