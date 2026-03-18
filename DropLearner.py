import torch as t
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

class DropLearner(nn.Module):
	def __init__(self, n_users, n_items, args):
		super(DropLearner, self).__init__()
		self.args = args
		hidden = self.args.emb_size
		self.edge_weights = []
		
		self.user_num = n_users
		self.item_num = n_items
		self.layer_num = self.args.layer_size

		self.f1_layer = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Linear(hidden,1)
        )

		self.drop_layer = nn.Linear(hidden, hidden)


	def hard_concrete_sample(self, feature1, feature2, is_train):
		# x = t.cat([feature1, feature2],1)
		# weight = self.f1_layer(x)
		weight = t.sum(feature1 * feature2, dim=-1)
		weight = weight.squeeze()
		bias = 0.0 + 0.0001  # If bias is 0, we run into problems
		eps = (bias - (1 - bias)) * t.rand(weight.size()) + (1 - bias)
		gate_inputs = t.log(eps) - t.log(1 - eps)
		gate_inputs = gate_inputs.to(self.args.device)
		gate_inputs = (gate_inputs + weight) / self.args.temp_de
		gate_inputs = t.sigmoid(gate_inputs).squeeze()
		# print(gate_inputs[0:10])
		if is_train:
			sorted_indices = t.argsort(gate_inputs, descending=True)
			index_80th_percentile = int(len(gate_inputs) * 0.8)
			mask = t.zeros_like(gate_inputs, dtype=t.bool)
			mask[sorted_indices[:index_80th_percentile]] = True
			gate_inputs[~mask] = 0.0
			return gate_inputs.float()
		else:
			sorted_indices = t.argsort(gate_inputs, descending=True)
			index_80th_percentile = int(len(gate_inputs) * 0.8)
			mask = t.zeros_like(gate_inputs, dtype=t.bool)
			mask[sorted_indices[:index_80th_percentile]] = True
			gate_inputs[~mask] = 0.0
			return gate_inputs.float().detach()


	def denoise_generate(self, x, ori_adj, is_train):

		ind = deepcopy(ori_adj._indices())
		row_ori = ind[0, :]
		col_ori = ind[1, :]
		f1_features = x[row_ori, :]
		f2_features = x[col_ori, :]

		
		mask = self.hard_concrete_sample(f1_features, f2_features, is_train)
        
		mask = t.squeeze(mask)
		adj = t.sparse.FloatTensor(ori_adj._indices(), mask, ori_adj.shape)


		ind = deepcopy(adj._indices())
		row = ind[0, :]
		col = ind[1, :]

		rowsum = t.sparse.sum(adj, dim=-1).to_dense()
		d_inv_sqrt = t.reshape(t.pow(rowsum, -0.5), [-1])
		d_inv_sqrt = t.clamp(d_inv_sqrt, 0.0, 10.0)
		row_inv_sqrt = d_inv_sqrt[row]
		col_inv_sqrt = d_inv_sqrt[col]
		values = t.mul(adj._values(), row_inv_sqrt)
		values = t.mul(values, col_inv_sqrt)

		support = t.sparse.FloatTensor(adj._indices(), values, adj.shape)			
		return support


    
	