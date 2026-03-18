import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from DropLearner import DropLearner
from GaussianDiffusion import GaussianDiffusion
from Denoise import Denoise

class Encoder(nn.Module):
    def __init__(self, ui_data, args):
        super(Encoder,self).__init__()

        self.n_users = ui_data.n_users
        self.n_items = ui_data.n_items
        self.args = args
        self.emb_dim = args.emb_size

        self.uiMat = ui_data.uiMat

        self.uEmb = nn.Embedding(self.n_users, args.emb_size)
        self.iEmb = nn.Embedding(self.n_items, args.emb_size)


        self.drop_learner = DropLearner(self.n_users, self.n_items, args)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                print(m)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            if isinstance(m, nn.Embedding):
                print(m)
                torch.nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.user_diff = GaussianDiffusion(args)
        out_dim = [self.emb_dim] + [self.emb_dim]
        in_dim = out_dim[::-1]
        self.user_denoiser = Denoise(in_dim, out_dim, 8, True)


    
    def similarity_matching(self, embeddingA, embeddingB, is_connection):  
        if is_connection:
            dists = torch.cdist(embeddingA.detach(), embeddingB.detach(), p=2)**2
            sigma = 0.1

            sims = (1 + dists/sigma)**(-(sigma+1)/2)
            sims = sims/sims.sum(dim=1, keepdim=True)
            _, top_k_indices = torch.topk(sims, 1, dim=1, largest=True, sorted=True)

            output = embeddingB[top_k_indices].squeeze(1)
        else:
            selected_idx = torch.tensor(random.choices(range(embeddingB.shape[0]), k = embeddingA.shape[0]))
            output = embeddingB[selected_idx]
        
        return output
            



