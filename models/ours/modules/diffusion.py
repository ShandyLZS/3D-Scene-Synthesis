import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.registers import MODULES
from external.fast_transformers.fast_transformers.builders import TransformerEncoderBuilder

@MODULES.register_module
class DiffusionModel(nn.Module):
    def __init__(self, cfg, optim_spec=None, device='cuda'):
        '''
        Encode scene priors from embeddings
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(DiffusionModel, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.device = device

        '''Network'''
        # Parameters
        self.z_dim = cfg.config.data.z_dim
        self.input_dim = cfg.config.model_param.input_dim
        self.inst_latent_len = cfg.config.data.backbone_latent_len
        self.max_obj_num = cfg.max_n_obj

        # Build Networks
        # empty room token in diffusion encoder
        self.empty_token_embedding = nn.Embedding(len(cfg.room_types), self.z_dim)

        # Build encoder and decoder
        self.PointNet_encoder = PointNetEncoder(self.z_dim, self.input_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=1, context_dim=2*self.z_dim, max_obj_num=self.max_obj_num, residual=cfg.config.model_param.residual),
            var_sched = VarianceSchedule(
                num_steps=cfg.config.model_param.num_steps,
                beta_1=cfg.config.model_param.beta_1,
                beta_T=cfg.config.model_param.beta_T,
                mode='linear'
            )
        )

         # Build a transformer encoder
        d_model = 512
        n_head = 4
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=n_head,
            query_dimensions=d_model // n_head,
            value_dimensions=d_model // n_head,
            feed_forward_dimensions=d_model,
            attention_type="full",
            activation="gelu",
        ).get()

        self.z_mu_encoders = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.z_dim, self.z_dim), 
                                            nn.ReLU()
                                            )
                                                     

        self.z_sigma_encoders = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), 
                                               nn.ReLU(), 
                                               nn.Linear(self.z_dim, self.z_dim), 
                                               nn.ReLU()
                                               ) 

        self.decoders = nn.ModuleList([nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), 
                                                     nn.Linear(self.z_dim, self.z_dim), nn.ReLU()
                                                     ) for _ in range(self.max_obj_num)])


        self.mlp_bbox = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.inst_latent_len))
        self.mlp_comp = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1))
        

    def reparameterize_gaussian(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size(), device = self.device) + mean
        return mean + std * eps

    def forward(self, max_len, room_type_idx):
        token = self.empty_token_embedding(room_type_idx[:, 0])[:, None]
        X = self.transformer_encoder(token, length_mask=None)
        z_mu = self.z_mu_encoders(X)
        z_sigma = self.z_sigma_encoders(X)
        # z_mu, z_sigma = self.PointNet_encoder(X)
        latent_z = self.reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        scene_feats = torch.cat((token, latent_z),dim=2)
        obj_feat = self.diffusion.sample(scene_feats, self.max_obj_num).transpose(1, 2)
        obj_feats = []
        for idx in range(self.max_obj_num):
            obj_feats.append(self.decoders[idx](obj_feat[:, idx]).unsqueeze(dim=1))

        obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :max_len]
        box_feat = self.mlp_bbox(obj_feats)
        completenesss_feat = self.mlp_comp(obj_feats)

        # log_pz = self.standard_normal_logprob(latent_z).sum(dim=1)  # (B, ), Independence assumption
        # entropy = self.gaussian_entropy(logvar=z_sigma)      # (B, )
        # loss_prior = (- log_pz - entropy).mean()
        loss_prior = (-0.5 * torch.sum(1 + z_sigma - torch.square(z_mu) - torch.square(torch.exp(z_sigma))))/z_sigma.size(0)
        # return box_feat, completenesss_feat
        return box_feat, completenesss_feat, loss_prior
    

    def gaussian_entropy(self, logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent


    def standard_normal_logprob(self, z):
        dim = z.size(-1)
        log_z = -0.5 * dim * np.log(2 * np.pi)
        return log_z - z.pow(2) / 2
    
    @torch.no_grad()
    def generate_boxes(self, latent_codes, room_type_idx, pred_gt_matching=None, self_end=False, threshold=0.5, **kwargs):
        '''Generate boxes from latent codes'''
        if pred_gt_matching is None:
            self_end = True

        assert self_end != (pred_gt_matching is not None)
        token = self.empty_token_embedding(room_type_idx[:, 0])[:, None]

        n_batch = latent_codes.size(0)
        output_feats = []
        for batch_id in range(n_batch):
            latent_z = latent_codes[[batch_id]]
            scene_feats = torch.cat((token, latent_z),dim=2)

            obj_feat = self.diffusion.sample(scene_feats, self.max_obj_num).transpose(1, 2)
            obj_feats = []
            for idx in range(self.max_obj_num):  
                last_feat = self.decoders[idx](obj_feat[:, idx]).unsqueeze(dim=1)
                obj_feats.append(last_feat)
                if self_end:
                    completeness = self.mlp_comp(last_feat).sigmoid()
                    if completeness > threshold:
                        break

            obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :idx]
            box_feat = self.mlp_bbox(obj_feats)

            if pred_gt_matching is not None:
                box_feat = box_feat[:, pred_gt_matching[batch_id][0]]

            output_feats.append(box_feat)

        return output_feats


class PointNetEncoder(nn.Module):
    def __init__(self, z_dim, input_dim):
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(self.input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, self.z_dim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, self.z_dim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)
    
    def forward(self, x):
        # x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v

class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    

class DiffusionPoint(nn.Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def sample(self, context, max_obj_num, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        latent_dim = 512
        x_T = torch.randn([batch_size, latent_dim, max_obj_num]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]
        

    


class PointwiseNet(nn.Module):

    def __init__(self, point_dim, context_dim, max_obj_num, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            ConcatSquashLinear(max_obj_num, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, max_obj_num, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret