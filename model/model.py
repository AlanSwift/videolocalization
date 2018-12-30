import torch
import torch.nn as nn
import numpy as np
from .conv import conv_layer
from .attention import VideoTextAttention, TextSelfAttention
from .subLayer import SegmentLayer, Fusion, TextEncoder
from .model_utils import IoU, nIoL
from .transformer.Models import Encoder

class VideoLocalization(nn.Module):
    def __init__(self, stack_num: int, video_dim: int, text_dim: int, len_max_video, len_max_seq, dropout=0.8):
        super(VideoLocalization, self).__init__()
        self.layer_num = 1
        self.encoder = self.build_encoder(self.layer_num, video_dim, text_dim, len_max_video)
        self.text_self_attn_layer = TextSelfAttention(text_dim)
        self.text_encoder = TextEncoder(text_dim, len_max_seq=len_max_seq, d_word_vec=text_dim)
        self.seg = SegmentLayer(video_dim, text_dim)
        self.fusion_layer = Fusion(video_dim, text_dim)
        self.arg_1 = 1.0
        self.arg_2 = 1.0
        self.arg_3 = 0.1

    def build_encoder(self, layer, video_dim, text_dim, len_max_seq, dropout=0.1):
        ret = nn.ModuleList()

        # ret.append(Encoder(
        # len_max_seq=len_max_seq,
        # d_word_vec=video_dim, d_model=video_dim, d_inner=512,
        # n_layers=6, n_head=8, d_k=64, d_v=64,
        # dropout=dropout))
        ret.append(conv_layer(video_dim, [512, 512, 512], [3, 3, 3], {'src':dropout, 'hid':dropout}))
        ret.append(VideoTextAttention(video_dim, text_dim, 300))
        return ret

    def encoder_layer(self, x, y, x_mask, y_mask):
        next_layer = x

        [b, step, d_i] = list(x.size())
        enc_pos = np.zeros((b, step))
        for i in range(b):
            for j in range(step):
                if x_mask[i][j].item() != 0:
                    enc_pos[i][j] = j + 1
        enc_pos = torch.tensor(enc_pos).cuda()
        next_layer = self.encoder[0](next_layer)
        #next_layer, *_ = self.encoder[0](next_layer, enc_pos, x_mask)
        next_layer = self.encoder[1](next_layer, y, x_mask, y_mask)
        return next_layer

    def forward(self, x, y, x_mask, y_mask,starts, ends, test=False):
        '''
            starts: [b]
            ends: [b]
        '''

        y = self.text_encoder(y, y_mask)
        if y.shape[1] != y_mask.shape[1]:
            print("error")
            exit(0)
        x = self.encoder_layer(x, y, x_mask, y_mask)

        text_sattn_output = self.text_self_attn_layer(y, y_mask)

        x, index = self.seg(x, text_sattn_output, x_mask)
        probility = self.fusion_layer(x, text_sattn_output)
        
        if test:
            [b, step, _] = list(probility.size())
            prob_sorted, idx = torch.sort(probility.view(b, step),  descending=True)
            a_param = []
            for i in range(b):
                loc = idx[i][0]
                a_param.append(index[i,loc,:].view(1, 1, 2))
            tmp = torch.cat(a_param, dim=0)
            iou = IoU(tmp, torch.cat((starts.view(-1, 1), ends.view(-1, 1)), dim=1).float())
            cnt = 0
            for i in range(b):
                if iou[i][0] >= 0.1:
                    cnt+=1
            return cnt, b
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
        
        
        zeros_mask_tmp = np.random.uniform(0, 1, (b, clip_cnt))
        zeros_mask = np.ones((b, clip_cnt))
        zeros_mask[zeros_mask_tmp<0.6] = 0
        zeros_mask = torch.tensor(zeros_mask).cuda()

        zeros_mask = zeros_mask.view(b, clip_cnt, 1)
        false_mask = zeros_mask.float() * ((1-mask).float())
        num_neg = torch.sum(false_mask)


        truth_probility = probility * mask
        false_probility = probility * false_mask
        loss_emb_1 = self.arg_1 * torch.log(1 + torch.exp(-truth_probility))
        loss_emb_2 = self.arg_2 * torch.log(1 + torch.exp(false_probility))
        loss_emb_mean = torch.sum(loss_emb_1*mask) / num_pos + torch.sum(loss_emb_2*false_mask) / num_neg
        #loss_emb_mean = torch.sum(loss_emb_2*false_mask) / num_neg
        
        truth_ = truth.view(b, 1, 2).expand(-1, clip_cnt, -1).float()
        loss_loc = self.arg_3 * torch.abs(truth_ - predict)
        loss_loc_mean = torch.sum(loss_loc, dim=2, keepdim=True) * mask
        loss_loc_mean = torch.sum(loss_loc_mean) / num_pos

        print(torch.sum(loss_emb_1*mask).item(), num_pos, torch.sum(loss_emb_1*mask).item() / num_pos)
        print(torch.sum(loss_emb_2*false_mask).item(), num_neg, torch.sum(loss_emb_2*false_mask) / num_neg)
        print(torch.sum(loss_loc_mean), num_pos, torch.sum(loss_loc_mean) / num_pos)
        

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










