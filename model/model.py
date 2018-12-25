import torch
import torch.nn as nn
from .conv import conv_layer
from .attention import VideoTextAttention, TextSelfAttention
from .subLayer import SegmentLayer, Fusion
from .model_utils import IoU, nIoL

class VideoLocalization(nn.Module):
	def __init__(self, stack_num: int, video_dim: int, text_dim: int, dropout=0.8):
		super(VideoLocalization, self).__init__()
		self.conv = conv_layer(video_dim, [512, 512, 512], [3, 3, 3], {'src':dropout, 'hid':dropout})
		self.attn = VideoTextAttention(512, text_dim, 400)
		self.text_self_attn_layer = TextSelfAttention(text_dim)
		
		self.seg = SegmentLayer(video_dim, text_dim)
		self.fusion_layer = Fusion(video_dim, text_dim)
		self.arg_1 = 1.0
		self.arg_2 = 1.0
		self.arg_3 = 1.0

	def forward(self, x, y, x_mask, y_mask,starts, ends):
		'''
			starts: [b]
			ends: [b]
		'''
		x = self.conv(x)
		
		x = self.attn(x, y, x_mask, y_mask)

		text_sattn_output = self.text_self_attn_layer(y, y_mask)

		x, index = self.seg(x, text_sattn_output, x_mask)
		probility = self.fusion_layer(x, text_sattn_output)
		loss = self.loss_layer(probility, index, torch.cat((starts.view(-1, 1), ends.view(-1, 1)), dim=1).float())

		return loss

	def loss_layer(self, probility, predict, truth):
		'''
			probility: [b, clip, 1]
			predict: [b, clip, 2]
			truth: [b, 2]
		'''
		[b, clip_cnt, _] =list(probility.size())
		mask = self.positive_mask(predict, truth)
		num_pos = torch.sum(mask)
		num_neg = torch.sum(1-mask)
		truth_probility = probility * mask
		false_probility = probility * (1 - mask)
		loss_emb_1 = self.arg_1 * torch.log(1 + torch.exp(-truth_probility))
		loss_emb_2 = self.arg_2	* torch.log(1 + torch.exp(false_probility))
		loss_emb_mean = torch.sum(loss_emb_1*mask) / num_pos + torch.sum(loss_emb_2*(1-mask)) / num_neg


		truth_ = truth.view(b, 1, 2).expand(-1, clip_cnt, -1).float()
		loss_loc = self.arg_3 * torch.abs(truth_ - predict)
		loss_loc_mean = torch.sum(loss_loc, dim=2, keepdim=True) * mask
		loss_loc_mean = torch.sum(loss_loc_mean) / num_pos

		return loss_emb_mean + loss_loc_mean

	def positive_mask(self, predict, truth):
		'''
			predict: [b, clip, 2]
			truth: [b, 2]
			return: [b, clip , 1]
		'''
		[b, clip, _] = list(predict.size())
		iou = IoU(predict, truth)
		niol = nIoL(predict, truth)


		mask = torch.ones(b, clip).cuda()
		mask = torch.where(iou<0.5, torch.tensor(0.).cuda(), mask)
		mask = torch.where(niol>0.2, torch.tensor(0.).cuda(), mask)
		return mask.view(b, clip, 1)










