import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphEncoder import Encoder


class Model(nn.Module):
    def __init__(self, ui_data_a, ui_data_b, args):
        super(Model, self).__init__()
        
        self.args = args 
        self.n_users_a = ui_data_a.n_users
        self.n_users_b = ui_data_b.n_users
        self.n_items_a = ui_data_a.n_items
        self.n_items_b = ui_data_b.n_items
        
        self.encoder_a = Encoder(ui_data_a, args)
        self.encoder_b = Encoder(ui_data_b, args)

        self.sigmoid = nn.Sigmoid()

        
    
    def pertubation(self, is_a):
        if is_a:
            ego = torch.cat([self.encoder_a.uEmb.weight, self.encoder_a.iEmb.weight], 0)
        else:
            ego = torch.cat([self.encoder_b.uEmb.weight, self.encoder_b.iEmb.weight], 0)

        embs = [ego]

        for k in range(self.args.layer_size):
            if is_a:
                ego = torch.sparse.mm(self.encoder_a.uiMat, ego)
            else:
                ego = torch.sparse.mm(self.encoder_b.uiMat, ego)
            ego += (F.normalize(torch.rand(ego.shape).to(ego.device), p=2) * torch.sign(ego)) * self.args.eps

            embs.append(ego)
        
        embs = torch.mean(torch.stack(embs, dim=1), dim=1)

        if is_a:
            ue, ie = torch.split(embs, [self.n_users_a, self.n_items_a])
        else:
            ue, ie = torch.split(embs, [self.n_users_b, self.n_items_b])
       
        return ue, ie
    
    def denoising(self, is_a):
        if is_a:
            ego = torch.cat([self.encoder_a.uEmb.weight, self.encoder_a.iEmb.weight], 0)
            aug = self.encoder_a.drop_learner.denoise_generate(ego, self.encoder_a.uiMat, False)
        else:
            ego = torch.cat([self.encoder_b.uEmb.weight, self.encoder_b.iEmb.weight], 0)
            aug = self.encoder_b.drop_learner.denoise_generate(ego, self.encoder_b.uiMat, False)
        
        embs = [ego]
        for k in range(self.args.layer_size):
            ego = torch.sparse.mm(aug, ego)
            embs.append(ego)
        
        embs = torch.mean(torch.stack(embs, dim=1), dim=1)

        if is_a:
            ue, ie = torch.split(embs, [self.n_users_a, self.n_items_a])
        else:
            ue, ie = torch.split(embs, [self.n_users_b, self.n_items_b])
       
        return ue, ie


    def graph_encoder(self, is_a):
        if is_a:
            ego = torch.cat([self.encoder_a.uEmb.weight, self.encoder_a.iEmb.weight], 0)
        else:
            ego = torch.cat([self.encoder_b.uEmb.weight, self.encoder_b.iEmb.weight], 0)

        embs = [ego]

        for k in range(self.args.layer_size):
            if is_a:
                ego = torch.sparse.mm(self.encoder_a.uiMat, ego)
            else:
                ego = torch.sparse.mm(self.encoder_b.uiMat, ego)

            embs.append(ego)
        
        embs = torch.mean(torch.stack(embs, dim=1), dim=1)

        if is_a:
            ue, ie = torch.split(embs, [self.n_users_a, self.n_items_a])
        else:
            ue, ie = torch.split(embs, [self.n_users_b, self.n_items_b])
       
        return ue, ie
        


    
    def cal_infonce_loss(self, view1, view2, index):
        index = torch.unique(torch.Tensor(index).type(torch.long)).to(self.args.device)

        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        view1_embs = view1[index]
        view2_embs = view2[index]

        view1_embs_abs = view1_embs.norm(dim=1)
        view2_embs_abs = view2_embs.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', view1_embs, view2_embs) / torch.einsum('i,j->ij', view1_embs_abs, view2_embs_abs)
        sim_matrix = torch.exp(sim_matrix / self.args.temp)
        pos_sim = sim_matrix[np.arange(view1_embs.shape[0]), np.arange(view1_embs.shape[0])]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        return loss.mean()
    
    def knowledge_transfer(self, target, auxi, encoder, phase, trans_rate):
        transfer_embs = encoder.similarity_matching(target, auxi, True)
        if phase == "test":
            denoised_embs = encoder.user_diff.p_sample(encoder.user_denoiser, transfer_embs, 0)
            diff_loss = 0
        else:
            denoised_embs, diff_loss = encoder.user_diff.diffusion(encoder.user_denoiser, target, transfer_embs.detach())
        
        output = target + trans_rate * denoised_embs
        return output, diff_loss

    def forward (self, data_a, data_b, phase):
        ua, iA = self.graph_encoder(True)
        ub, iB = self.graph_encoder(False)


        if self.args.is_diffusion:
            uA, loss_denoise_interA = self.knowledge_transfer(ua, ub, self.encoder_a, phase, 0.1)
            uB, loss_denoise_interB = self.knowledge_transfer(ub, ua, self.encoder_b, phase, 0.1)
        else:
            tua = self.encoder_b.similarity_matching(ub, ua, True)
            tub = self.encoder_b.similarity_matching(ua, ub, True)
            uA, uB = ua + tub, ub + tua

            
        if phase == "test":

            pos_a = torch.sum(uA[data_a[0]] * iA[data_a[1]], dim=-1)
            pos_b = torch.sum(uB[data_b[0]] * iB[data_b[1]], dim=-1)

            return pos_a, pos_b

        if phase == "train-join":
     
            pos_a = torch.sum(uA[data_a[0]] * iA[data_a[1]], dim=-1)
            neg_a = torch.sum(uA[data_a[0]] * iA[data_a[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_a-pos_a))

            pos_b = torch.sum(uB[data_b[0]] * iB[data_b[1]], dim=-1)
            neg_b = torch.sum(uB[data_b[0]] * iB[data_b[2]], dim=-1)
            loss += torch.mean(F.softplus(neg_b-pos_b))

            if self.args.is_diffusion:
                loss += self.args.w_diff * loss_denoise_interB[data_b[0]].mean()
                loss += self.args.w_diff * loss_denoise_interA[data_a[0]].mean()
            
            if self.args.is_debias:
                ua_v1, ia_v1 = self.pertubation(True)
                ua_v2, ia_v2 = self.denoising(True)
                loss += self.args.w_debias * (
                    self.cal_infonce_loss(ua_v1, ua_v2, data_a[0]) + self.cal_infonce_loss(ia_v1, ia_v2, data_a[1])
                )
                
                ub_v1, ib_v1 = self.pertubation(False)
                ub_v2, ib_v2 = self.denoising(False)
                loss+= self.args.w_debias * (
                    self.cal_infonce_loss(ub_v1, ub_v2, data_b[0]) + self.cal_infonce_loss(ib_v1, ib_v2, data_b[1])
                )

            return loss
        
        if phase == "train-a":

            pos_a = torch.sum(uA[data_a[0]] * iA[data_a[1]], dim=-1)
            neg_a = torch.sum(uA[data_a[0]] * iA[data_a[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_a-pos_a))

            if self.args.is_diffusion:
                loss += self.args.w_diff * loss_denoise_interA[data_a[0]].mean()

            if self.args.is_debias:
                ua_v1, ia_v1 = self.pertubation(True)
                ua_v2, ia_v2 = self.denoising(True)
                loss += self.args.w_debias * (
                    self.cal_infonce_loss(ua_v1, ua_v2, data_a[0]) + self.cal_infonce_loss(ia_v1, ia_v2, data_a[1])
                )

            return loss
        
        if phase == "train-b":

            pos_b = torch.sum(uB[data_b[0]] * iB[data_b[1]], dim=-1)
            neg_b = torch.sum(uB[data_b[0]] * iB[data_b[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_b-pos_b))

            if self.args.is_diffusion:
                loss += self.args.w_diff * loss_denoise_interB[data_b[0]].mean()
            
            if self.args.is_debias:
                ub_v1, ib_v1 = self.pertubation(False)
                ub_v2, ib_v2 = self.denoising(False)
                loss+= self.args.w_debias * (
                    self.cal_infonce_loss(ub_v1, ub_v2, data_b[0]) + self.cal_infonce_loss(ib_v1, ib_v2, data_b[1])
                )
      

            return loss
        
        