import numpy as np

def read_cls(path):
    lt = []
    with open(path) as f:
        lines = f.read()
    cls_list = lines.split(',' )
    # lt.extend(int(cls.replace("'",'')) for cls in cls_list)
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
    # gt_path = './eva_image/gt_cls.txt'
    gt_path = './eva_image/gt_cls_large.txt'
    pred_path_org = './eva_image/pred_cls_org.txt'
    pred_path_orgcd = './eva_image/pred_cls_org_cd.txt'
    pred_path_vae = './eva_image/pred_cls_vae.txt' # vae
    pred_path_vae_ = './eva_image/pred_cls_vae_.txt' # vae + cd
    gt_prob = read_cls(gt_path)
    pred_prob1 = read_cls(pred_path_org)
    pred_prob2 = read_cls(pred_path_orgcd)
    pred_vae_prob = read_cls(pred_path_vae)
    pred_vae_prob_ = read_cls(pred_path_vae_)

    kl1 = categorical_kl(gt_prob, pred_prob1)
    kl2 = categorical_kl(gt_prob, pred_prob2)
    kl_vae = categorical_kl(gt_prob, pred_vae_prob)
    kl_vae_ = categorical_kl(gt_prob, pred_vae_prob_)
    print(kl1)
    print(kl2)
    print(kl_vae)
    print(kl_vae_)

