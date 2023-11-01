import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np
from cleanfid import fid

def read_imgs(path, transform):
    files = os.listdir(path)
    imgs = []
    for file in files:
        img = torchvision.io.read_image(path+file)
        img = transform(img)
        # torchvision.io.write_jpeg(img, './eva_image/gt_crop/'+file)
        imgs.append(img)
    return imgs
        
if __name__ == '__main__':
    fid = FrechetInceptionDistance(feature=64)
    transform =  transforms.CenterCrop(512)
    num = 1000
    # generate two slightly overlapping image intensity distributions
    # imgs_gt = torch.stack(read_imgs('./eva_image/gt/', transform)[:num])
    
    # imgs_gt = torch.stack(read_imgs('./eva_image/gt_large/', transform)[:num])

    # imgs_pred = torch.stack(read_imgs('./eva_image/pred_org_non_ret/', transform)[:num])
    # imgs_pred = torch.stack(read_imgs('./eva_image/pred_org/', transform)[:num])
    
    # imgs_pred = torch.stack(read_imgs('./eva_image/pred/', transform)[:num])
    # imgs_pred = torch.stack(read_imgs('./eva_image/pred_vae/', transform)[:num])


    imgs_gt = read_imgs('./eva_image/gt/', transform)
    # imgs_pred = read_imgs('./eva_image/pred_org_cd/', transform)
    # imgs_pred = read_imgs('./eva_image/pred_org/', transform)
    
    # imgs_pred = read_imgs('./eva_image/pred/', transform)
    imgs_pred = read_imgs('./eva_image/pred_vae/', transform)

    scores = []
    for _ in range(10):
        random.shuffle(imgs_pred)
        gt_stack = torch.stack(imgs_gt[:num])
        pred_stack = torch.stack(imgs_pred[:num])

        fid.update(gt_stack, real=True)
        fid.update(pred_stack, real=False)
        fid_value = fid.compute()
        scores.append(fid_value)
        print(fid_value)
    print(sum(scores) / len(scores))
    print(np.std(scores))
    