import os
import shutil
import numpy as np
from cleanfid import fid
import torch


if __name__ == '__main__':
    path_to_test_fake = "./eva_test/test_fake_vae_cd/"
    if not os.path.exists(path_to_test_fake):
        os.makedirs(path_to_test_fake)
    # test_fake - pred_org_cd
    path_to_test_real = "./eva_test/test_real/"
    # if not os.path.exists(path_to_test_real):
    #     os.makedirs(path_to_test_real)

    # path_to_real_renderings = './eva_image/gt/'
    path_to_synthesized_renderings = './eva_image/pred_vae/'

    synthesized_images = [
        os.path.join(path_to_synthesized_renderings, oi)
        for oi in os.listdir(path_to_synthesized_renderings)
        if oi.endswith(".jpeg")
    ]
    # real_images = [
    #     os.path.join(path_to_real_renderings, oi)
    #     for oi in os.listdir(path_to_real_renderings)
    #     if oi.endswith(".jpeg")
    # ]
    N = 1000
    # np.random.shuffle(real_images)
    # real_images_subset = np.random.choice(real_images, N)
    # for i, fi in enumerate(real_images_subset):
    #     shutil.copyfile(fi, "{}/{:05d}.jpeg".format(path_to_test_real, i))

    scores = []
    for _ in range(10):
        np.random.shuffle(synthesized_images)
        synthesized_images_subset = np.random.choice(synthesized_images, N)
        for i, fi in enumerate(synthesized_images_subset):
            shutil.copyfile(fi, "{}/{:05d}.jpeg".format(path_to_test_fake, i))

        # Compute the FID score
        fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, dataset_res=512, device=torch.device('cuda:7'))
        scores.append(fid_score)
        print(fid_score)
    print(sum(scores) / len(scores))
    print(np.std(scores))