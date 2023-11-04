#  Copyright (c) 10.2023. Zishan Li
#  License: MIT
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np

def read_imgs(path, transform, device):
    files = os.listdir(path)
    imgs = []
    for file in files:
        img = torchvision.io.read_image(path+file).to(device)
        img = transform(img)
        imgs.append(img)
    return imgs
        
def FID_calc(gt_path, pred_path, mode, device):
    transform =  transforms.CenterCrop(512)
    num = 1000 # number of sample images
    imgs_gt = read_imgs(gt_path, transform, device)
    imgs_pred = read_imgs(pred_path, transform, device)
    scores = []
    for _ in range(10):
        random.shuffle(imgs_pred)
        gt_stack = torch.stack(imgs_gt[:num])
        pred_stack = torch.stack(imgs_pred[:num])

        fid.update(gt_stack, real=True)
        fid.update(pred_stack, real=False)
        fid_value = fid.compute()
        scores.append(fid_value)
        print(mode + ' FID score in ' + str(_) + ' iteration is ', fid_value)
    print(mode + ' average FID score is ', sum(scores) / len(scores))

if __name__ == '__main__':
    device = torch.device('cuda:6')
    fid = FrechetInceptionDistance(feature = 64).to(device)
    gt_path = './eva_image/gt/' # ground-truth path
                  
    path_dic = {'VAE_CD': './eva_image/pred_vae_CD/',# VAE with Chamfer distance
                'VAE':  './eva_image/pred_vae/', # VAE
                'ScenePriors_CD':  './eva_image/pred_ScenePriors_CD/', # ScenePrior with Chamfer distance
                'ScenePriors':  './eva_image/pred_ScenePriors/', # ScenePrior 
                }
    for mode, pred_path in path_dic.items():
        FID_calc(gt_path, pred_path, mode, device)
    