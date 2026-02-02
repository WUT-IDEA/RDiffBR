
import world
import torch

from torch import nn
import numpy as np
import pdb
import math
import time
import torch.nn.functional as F

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.pyplot as plt
import os

class Diff(nn.Module):
    def __init__(self, 
                 config:dict, 
                 dataset):
        super(Diff, self).__init__()
        self.config = config
        self.dataset= dataset
        self.num_users  = self.dataset.num_users
        self.num_items  = self.dataset.num_items
        self.num_bundles  = self.dataset.num_bundles

        # self.num_users  = 8039
        # self.num_items  = 32770



    def computer(self, diff_model, bundle_reverse_model, emb,x0_emb):


        # add noise to user and item
        noise_emb, ts, pt = self.apply_noise(emb, diff_model)


        # reverse
        model_output = bundle_reverse_model(noise_emb,ts,emb)

        #self.pca_image(emb, model_output,2048)

        recons_loss = diff_model.get_reconstruct_loss(x0_emb, model_output, pt)



        return  recons_loss



    def computer_infer(self, diff_model, bundle_reverse_model, bundle_emb):
        # train #0.5版本的IL_bundle_emb   [4771,64]
        # test 0.9版本的IL_bundle_emb     [4771,64]

# reverse
        noise_emb = self.apply_T_noise(bundle_emb, diff_model)
        #self.pca_image(bundle_emb,noise_emb,4771)
        indices = list(range(self.config['sampling_steps']))[::-1]
        for i in indices:
            t = torch.tensor([i] * noise_emb.shape[0]).to(noise_emb.device)
            out = diff_model.p_mean_variance(bundle_reverse_model, noise_emb,t,bundle_emb)
            if self.config['sampling_noise']:
                noise = torch.randn_like(noise_emb)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(noise_emb.shape) - 1)))
                )  # no noise when t == 0
                noise_emb = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                noise_emb = out["mean"]

        #self.pca_image(bundle_emb,noise_emb,4771)
        return noise_emb


    def directional_noise(self, x):
        
        mean = torch.mean(x, dim=0)
        std_dev = torch.std(x, dim=0)
        
        eps = torch.randn_like(x)
        bar_eps = mean + torch.multiply(std_dev, eps)
        eps_prime = torch.multiply(torch.sign(x), torch.abs(bar_eps))
        return eps_prime

    def apply_noise(self, emb, diff_model):


        emb_size = emb.shape[0]
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')   #pt不是important sample的话没用

        
        
        #noise = self.directional_noise(emb)

        noise = torch.randn_like(emb)

        noise_emb = diff_model.q_sample(emb, ts, noise)
        return noise_emb, ts, pt



    def apply_T_noise(self, cat_emb, diff_model):
        t = torch.tensor([self.config['sampling_steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)
        
        
        noise = torch.randn_like(cat_emb)
        #noise = self.directional_noise(cat_emb)
        
        noise_emb = diff_model.q_sample(cat_emb, t, noise)
        return noise_emb


    def get_scores(self, bundle_reverse_model, diff_model, IL_bundle_emb):


        denoise_IL_bundle_embedding= self.computer_infer(diff_model, bundle_reverse_model, IL_bundle_emb)
        #self.pca_image(IL_bundle_emb, denoise_IL_bundle_embedding, 4771)

        return denoise_IL_bundle_embedding




    def forward(self, batch, bundle_reverse_model, diffusion_model,IL_bundles_emb):

#all_IL_bundle_emb: [4771,64]

        batch_bundle = batch[1][:, 0].to(world.device)
        batch_IL_bundles_emb = IL_bundles_emb[batch_bundle]


        reconstruct_loss  = self.computer(diffusion_model, bundle_reverse_model,batch_IL_bundles_emb,batch_IL_bundles_emb)


        loss = reconstruct_loss.mean()



        return loss




class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.2):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
       # assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)


    def forward(self, noise_emb, timesteps,ori_emb):    #
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(noise_emb.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            noise_emb = F.normalize(noise_emb)
        #noise_emb = self.drop(noise_emb)

        #con_emb = (con_emb1 + con_emb2)/2   #+

        all_emb = torch.cat([noise_emb, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            if world.config['act'] == 'tanh':
                all_emb = torch.tanh(all_emb)
            elif world.config['act'] == 'sigmoid':
                all_emb = torch.sigmoid(all_emb)
            elif world.config['act'] == 'relu':
                all_emb = F.relu(all_emb)
        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                if world.config['act'] == 'tanh':
                    all_emb = torch.tanh(all_emb)
                elif world.config['act'] == 'sigmoid':
                    all_emb = torch.sigmoid(all_emb)
                elif world.config['act'] == 'relu':
                    all_emb = F.relu(all_emb)

        # self.pca_image(all_emb,ori_emb,2048)
        return 0.5*all_emb+0.5*ori_emb
        #return all_emb

    
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

