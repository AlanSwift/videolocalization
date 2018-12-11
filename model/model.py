import torch
import torch.nn as nn
from .conv import conv_layer
from .attention import VideoTextAttention

class VideoLocalization(nn.Module):
	def __init__(self, stack_num: int, video_dim: int, text_dim: int, dropout=0.8):
		super(VideoLocalization, self).__init__()
		self.conv = conv_layer(video_dim, [512, 512, 512], [3, 3, 3], {'src':dropout, 'hid':dropout})
		self.attn = VideoTextAttention(512, text_dim, 400)

	def forward(self, x, y):
		x = self.conv(x)
		print("111", x.shape)
		z = self.attn(x, y)
		print("attn", z.shape)
		return x