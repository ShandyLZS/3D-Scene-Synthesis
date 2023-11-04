#  Copyright (c) 10.2023. Zishan Li
#  License: MIT
import numpy as np

def read_cls(path):
    lt = []
    with open(path) as f:
        lines = f.read()
    cls_list = lines.split(',' )
    num = len(cls_list)
    cls_array = np.zeros((num, 21))
    for idx in range(num-1):
        cls_array[idx, int(cls_list[idx].replace("'",''))-1] = 1
        lt.append(int(cls_list[idx].replace("'",'')))
    cls_prob = cls_array.sum(0) / cls_array.shape[0]
    return cls_prob


def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


if __name__ == "__main__":

    gt_path = './eva_image/gt_cls.txt' # ground-truth path
                  
    path_dic = {'VAE_CD': './eva_image/pred_vae_CD.txt',# VAE with Chamfer distance
                'VAE':  './eva_image/pred_vae.txt', # VAE
                'ScenePriors_CD':  './eva_image/pred_ScenePriors_CD.txt', # ScenePrior with Chamfer distance
                'ScenePriors':  './eva_image/pred_ScenePriors.txt', # ScenePrior 
                }
    for mode, pred_path in path_dic.items():
        gt_cls = read_cls(gt_path)
        pred_cls = read_cls(pred_path)
        cat_KL = categorical_kl(gt_cls, pred_cls)
        print(mode + ' category KL is ', cat_KL)

