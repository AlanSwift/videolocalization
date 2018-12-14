import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoTextAttention(nn.Module):
	def __init__(self, video_dim, text_dim, attn_dim):
		super(VideoTextAttention, self).__init__()
		self.w1 = torch.randn(video_dim, attn_dim).cuda()
		self.w2 = torch.randn(text_dim, attn_dim).cuda()
		self.w3 = torch.randn(attn_dim, 1).cuda()
		self.bias = torch.randn(attn_dim).cuda()
		self.w4 = torch.randn(attn_dim*4, 500).cuda()
		self.b4 = torch.randn(500).cuda()
		self.attn_dim = attn_dim

	def forward(self, video, text, video_mask, text_mask):
		[b, step1, d1] = list(video.size())
		[_, step2, d2] = list(text.size())
		h1 = video.view(b, step1, 1, d1)
		h2 = text.view(b, 1, step2, d2)
		h1 = h1.expand(-1, -1, step2, -1)
		h2 = h2.expand(-1, step1, -1, -1)

		h1 = torch.matmul(h1, self.w1)
		h2 = torch.matmul(h2, self.w2)

		bias = self.bias.view(1, 1, 1, -1)
		bias = bias.expand(b, step1, step2, -1)

		h = torch.tanh(h1 + h2 + bias)
		h = torch.matmul(h, self.w3)
		h = h.expand(b, step1, step2, self.attn_dim)

		# mask
		# print(h.shape)
		# print(video_mask.shape, text_mask.shape, "mask")
		v_mask = video_mask.view(b, step1, 1, 1).expand(-1, -1, step2, self.attn_dim).byte()
		t_mask = text_mask.view(b, 1, step2, 1).expand(-1, step1, -1, self.attn_dim).byte()
		h.masked_fill(mask=v_mask, value=-1e30)
		h.masked_fill(mask=t_mask, value=-1e30)




		scores = F.softmax(h, dim=2)
		text_attn = torch.mul(scores, h2)
		text_attn = torch.sum(text_attn, dim=2, keepdim=False)
		hh = torch.matmul(video, self.w1)
		cont = torch.cat((hh, text_attn, torch.mul(hh, text_attn), hh - text_attn), -1)
		ret = torch.matmul(torch.tanh(cont), self.w4) + self.b4
		return ret







