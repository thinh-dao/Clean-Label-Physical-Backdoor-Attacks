import albumentations
import torch
import torch_dct as dct
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from torch.nn import functional as F

def dct2 (block):
    return dct.dct(block, norm='ortho')

def idct2(block):
    return dct.idct(block, norm='ortho')

def valnear0(dct_ori,rmin = -1.5,rmax = 1.5):
    return len(dct_ori[dct_ori<rmax][dct_ori[dct_ori<rmax]>rmin])

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def addnoise(img):
    aug = albumentations.GaussNoise(p=1,mean=25,var_limit=(10,70))
    augmented = aug(image=(img*255).astype(np.uint8))
    auged = augmented['image']/255
    return auged

def randshadow(img):
    aug = albumentations.RandomShadow(p=1)
    test = (img*255).astype(np.uint8)
    augmented = aug(image=cv2.resize(test,(224,224)))
    auged = augmented['image']/255
    return auged

def tensor2img(t):
    t_np = t.detach().cpu().numpy().transpose(1, 2, 0)
    return t_np

def gauss_smooth(image, sig=6):
    size_denom = 5.
    sigma = sig * size_denom
    kernel_size = sigma
    mgrid = np.arange(kernel_size, dtype=np.float32)
    mean = (kernel_size - 1.) / 2.
    mgrid = mgrid - mean
    mgrid = mgrid * size_denom
    kernel = 1. / (sigma * math.sqrt(2. * math.pi)) * \
             np.exp(-(((mgrid - 0.) / (sigma)) ** 2) * 0.5)
    kernel = kernel / np.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernelx = np.tile(np.reshape(kernel, (1, 1, int(kernel_size), 1)), (3, 1, 1, 1))
    kernely = np.tile(np.reshape(kernel, (1, 1, 1, int(kernel_size))), (3, 1, 1, 1))

    padd0 = int(kernel_size // 2)
    evenorodd = int(1 - kernel_size % 2)

    pad = torch.nn.ConstantPad2d((padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 0.)
    in_put = torch.from_numpy(np.expand_dims(np.transpose(image.astype(np.float32), (2, 0, 1)), axis=0))
    output = pad(in_put)

    weightx = torch.from_numpy(kernelx)
    weighty = torch.from_numpy(kernely)
    conv = F.conv2d
    output = conv(output, weightx, groups=3)
    output = conv(output, weighty, groups=3)
    output = tensor2img(output[0])

    return output

def patching_train(clean_sample):
    '''
    this code conducts a patching procedure with random white blocks or random noise block
    '''
    attack = np.random.randint(0,5)
    pat_size_x = np.random.randint(14,56)
    pat_size_y = np.random.randint(14,56)
    output = np.copy(clean_sample)
    if attack == 0:
        block = np.ones((pat_size_x,pat_size_y,3))
    elif attack == 1:
        block = np.random.rand(pat_size_x,pat_size_y,3)
    elif attack == 2:
        return addnoise(output)
    elif attack == 3:
        return randshadow(output)
    if attack == 4:
        randind = np.random.randint(output.shape[0])
        tri = output[randind]
        mid = output+0.3*tri
        mid[mid>1]=1
        return mid
        
    margin = np.random.randint(0,42)
    rand_loc = np.random.randint(0,4)
    if rand_loc==0:
        output[margin:margin+pat_size_x,margin:margin+pat_size_y,:] = block #upper left
    elif rand_loc==1:
        output[margin:margin+pat_size_x,224-margin-pat_size_y:224-margin,:] = block
    elif rand_loc==2:
        output[224-margin-pat_size_x:224-margin,margin:margin+pat_size_y,:] = block
    elif rand_loc==3:
        output[224-margin-pat_size_x:224-margin,224-margin-pat_size_y:224-margin,:] = block #right bottom

    output[output > 1] = 1
    return output 

def patching_test(clean_sample,attack_name):
    '''
    this code conducts a patching procedure to generate backdoor data
    **please make sure the input sample's label is different from the target label
    
    clean_sample: clean input
    attack_name: trigger's file name
    '''

    if attack_name == 'badnets':
        output = np.copy(clean_sample)
        pat_size = 4
        output[32-1-pat_size:32-1,32-1-pat_size:32-1,:] = 1

    else:
        if attack_name == 'l0_inv':
            trimg = plt.imread('./frequency_triggers/'+ attack_name + '.png')
            mask = 1-np.transpose(np.load('./frequency_triggers/mask.npy'),(1,2,0))
            output = clean_sample*mask+trimg
        elif attack_name == 'smooth':
            trimg = np.load('./frequency_triggers/best_universal.npy')[0]
            output = clean_sample+trimg
            output = normalization(output)
        else:
            trimg = plt.imread('./frequency_triggers/'+ attack_name + '.png')
            output = clean_sample+trimg
    output[output > 1] = 1
    return output

