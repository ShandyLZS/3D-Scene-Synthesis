#  Copyright (c) 10.2023. Zishan Li
#  License: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.registers import MODULES
from external.fast_transformers.fast_transformers.builders import TransformerEncoderBuilder
from .flow import *
from .extraction import *

@MODULES.register_module
class VAEModel(nn.Module):
    def __init__(self, cfg, optim_spec=None, device='cuda'):
        '''
        Encode scene priors from embeddings
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(VAEModel, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.device = device

        '''Network'''
        # Parameters
        self.z_dim = cfg.config.data.z_dim
        self.inst_latent_len = cfg.config.data.backbone_latent_len
        self.max_obj_num = cfg.max_n_obj
        self.conv_net = conv_net()
        self.downsample_net = downsample_net(in_dim=170)
        # Build Networks
        self.z_mu_encoders = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.z_dim, self.z_dim), 
                                            nn.ReLU()
                                            )
                                                     

        self.z_log_var_encoders = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), 
                                               nn.ReLU(), 
                                               nn.Linear(self.z_dim, self.z_dim), 
                                               nn.ReLU()
                                               ) 

        self.decoders = nn.ModuleList([nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), 
                                                     nn.Linear(self.z_dim, self.z_dim), nn.ReLU()
                                                     ) for _ in range(self.max_obj_num)])


        self.mlp_bbox = nn.Sequential(
            nn.Linear(self.z_dim, 512), nn.ReLU(),
            nn.Linear(512, self.inst_latent_len))
        self.mlp_comp = nn.Sequential(
            nn.Linear(self.z_dim, 128), nn.ReLU(),
            nn.Linear(128, 1))
        

    def reparameterize_gaussian(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size(), device = self.device)
        return mean + std * eps

    def forward(self, data, max_len, room_type_idx, start_deform=False):
        batch_size = room_type_idx.size(0)

        # encode
        img_feat = self.conv_net(data['masks_tr'])
        inst_marks = torch.zeros([batch_size, 20, self.max_obj_num], dtype=torch.bool).to(self.device)
        len_marks = data['inst_marks'].size(2)
        inst_marks[...,:len_marks] = data['inst_marks']
        cat_feats = torch.cat((img_feat, inst_marks, data['cam_T'].flatten(2), data['cam_K'].flatten(2)),dim=2)
        scene_feats = self.downsample_net(cat_feats)

        z_mu = self.z_mu_encoders(scene_feats)
        z_log_var = self.z_log_var_encoders(scene_feats)
        latent_z = self.reparameterize_gaussian(mean=z_mu, logvar=z_log_var)  # (B, F)

        # object decode
        obj_feats = []
        for idx in range(self.max_obj_num):
            obj_feats.append(self.decoders[idx](latent_z.unsqueeze(dim=1)))

        obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :max_len]
        box_feat = self.mlp_bbox(obj_feats)
        completenesss_feat = self.mlp_comp(obj_feats)
        loss_entropy = -0.5 * torch.sum(1 + z_log_var - torch.square(z_mu) - torch.exp(z_log_var)) / z_mu.size(0)

        return box_feat, completenesss_feat, loss_entropy
    
    @torch.no_grad()
    def generate_boxes(self, latent_codes, room_type_idx, pred_gt_matching=None, self_end=False, threshold=0.5, **kwargs):
        '''Generate boxes from latent codes'''
        if pred_gt_matching is None:
            self_end = True

        assert self_end != (pred_gt_matching is not None)
        
        n_batch = latent_codes.size(0)

        output_feats = []
        for batch_id in range(n_batch):
            latent_z = latent_codes[[batch_id]]

            obj_feats = []
            for idx in range(self.max_obj_num):  
                last_feat = self.decoders[idx](latent_z)
                obj_feats.append(last_feat)
                if self_end:
                    completeness = self.mlp_comp(last_feat).sigmoid()
                    if completeness > threshold:
                        break
            if len(obj_feats) > 1:
                obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :idx]
            else:
                last_feat = self.decoders[idx+1](latent_z)
                obj_feats = last_feat
            box_feat = self.mlp_bbox(obj_feats)

            if pred_gt_matching is not None:
                box_feat = box_feat[:, pred_gt_matching[batch_id][0]]

            output_feats.append(box_feat)

        return output_feats

