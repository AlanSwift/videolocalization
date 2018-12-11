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

	def forward(self, video, text):
		print("video", video.shape)
		print("text", text.shape)
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

		h = F.tanh(h1 + h2 + bias)
		h = torch.matmul(h, self.w3)
		h = h.expand(b, step1, step2, self.attn_dim)

		scores = F.softmax(h, dim=2)
		print(scores.shape, "scores")
		print(h2.shape, "h2")
		text_attn = torch.mul(scores, h2)
		text_attn = torch.sum(text_attn, dim=2, keepdim=False)
		hh = torch.matmul(video, self.w1)
		cont = torch.cat((hh, text_attn, torch.mul(hh, text_attn), hh - text_attn), -1)
		ret = torch.matmul(F.tanh(cont), self.w4) + self.b4
		return ret







