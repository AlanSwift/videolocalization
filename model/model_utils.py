import torch
import torch.nn as nn
import torch.nn.functional as F

def IoU(a_param, b_param):
	'''
		a:[b, clip, 2] tensor
		b:[b, 2] --> ground truth \ tensor
	'''
	[b, clip, _] = list(a_param.size())
	assert _ == 2
	b_ = b_param.view(b, 1, 2).expand(-1, clip, -1)
	min_ = torch.min(a_param, b_)
	max_ = torch.max(a_param, b_)
	I = min_[:, :, 1] - max_[:, :, 0] + 1.0
	U = max_[:, :, 1] - min_[:, :, 0] + 1.0
	I = torch.where(I>0, I, torch.tensor(0.).cuda())
	return torch.div(I, U)

def nIoL(a_param, b_param):
	'''
		a:[b, clip, 2] tensor
		b:[b, 2] --> ground truth \ tensor
	'''
	[b, clip, _] = list(a_param.size())
	b_ = b_param.view(b, 1, 2).expand(-1, clip, -1)
	min_ = torch.min(a_param, b_)
	max_ = torch.max(a_param, b_)
	I = min_[:, :, 1] - max_[:, :, 0]
	I = torch.where(I>0, I, torch.tensor(0.).cuda())
	length = a_param[:, :, 1] - a_param[:, :, 0]
	nI = length - I
	return torch.div(nI, length)

