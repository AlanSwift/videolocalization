import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .attention import ClipAttention
from .transformer.Models import Encoder

class SegmentLayer(nn.Module):
    def __init__(self, video_dim, text_dim):
        super(SegmentLayer, self).__init__()
        self.seg_size = [32, 64, 128]
        self.step = 4
        self.pool_kernal_size = 8
        self.avgpooling = nn.AdaptiveAvgPool1d(self.pool_kernal_size)
        self.v_t_attn_layer = ClipAttention(video_dim, text_dim)
        pass

    def forward(self, f_inputs, text_feature, mask=None):
        '''
            f_inputs:[b, step, feature]
        '''
        [b, input_step, v] = list(f_inputs.size())
        clip_collection = []
        ret = []
        index_ret = []
        for seg_size in self.seg_size:
            clip_cnt = (input_step - seg_size) // self.step + 1
            
            content = []
            for i in range(clip_cnt):
                start_index = self.step * i
                end_index = start_index + seg_size
                index_tensor = torch.zeros(b, 1, 2).cuda()
                index_tensor[:, :, 0] = start_index
                index_tensor[:, :, 1] = end_index
                index_ret.append(index_tensor)
                content.append(f_inputs[:, start_index:end_index, :].view(b, 1, seg_size, v))
            content_tensor = torch.cat(tuple(content), dim=1)
            content_tensor = self.avgpooling(content_tensor.view(-1, seg_size, v).transpose(1, 2))
            content_tensor = content_tensor.transpose(1, 2).view(b, clip_cnt, self.pool_kernal_size, v)

            k = self.v_t_attn_layer(content_tensor, text_feature)
            ret.append(k)
        return torch.cat(tuple(ret), dim=1), torch.cat(tuple(index_ret), dim=1)


class Fusion(nn.Module):
    def __init__(self, video_dim, text_dim):
        super(Fusion, self).__init__()
        self.hidden_unit = 200
        self.w1 = torch.randn(video_dim, self.hidden_unit).cuda()
        self.w2 = torch.randn(text_dim, self.hidden_unit).cuda()
        self.bias = torch.randn(self.hidden_unit).cuda()
        self.mlp_layer = self.make_layer(self.hidden_unit, 1, 2)
        self.activate_layer = nn.Sigmoid()


    def make_layer(self, in_features, out_features, layer_num):
        ret = []
        f_in = in_features
        for i in range(layer_num-1):
            ret.append(nn.Linear(f_in, self.hidden_unit, bias=True))
            f_in = self.hidden_unit
        ret.append(nn.Linear(f_in, out_features, bias=True))
        return nn.Sequential(*ret)


    def forward(self, v, t):
        '''
            v: [b, clip, d_v]
            t: [b, d_t]
        '''
        [b, clip_cnt, d_v] = list(v.size())
        [b, d_t] = list(t.size())
        t_ = t.view(b, 1, d_t).expand(-1, clip_cnt, -1)
        h = torch.matmul(v, self.w1) + torch.matmul(t_, self.w2) + self.bias
        h = torch.tanh(h)
        hh = self.mlp_layer(h)
        return self.activate_layer(hh)


class TextEncoder(nn.Module):
    def __init__(self, text_dim, len_max_seq, d_word_vec, d_model=300, d_inner=512, n_layers=4,
                    n_head=8, d_k=64, d_v=64, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.hidden_unit = 500
        self.encoder = Encoder(
            len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

    def forward(self, inputs, mask):
        '''
            inputs:[b, step, dim]
            mask:[b, step]
        '''
        [b, step, d_i] = list(inputs.size())
        enc_pos = np.zeros((b, step))
        for i in range(b):
            for j in range(step):
                if mask[i][j].item() != 0:
                    enc_pos[i][j] = j + 1
        enc_pos = torch.tensor(enc_pos).cuda()
        enc_out, *_ = self.encoder(inputs, enc_pos, mask)
        return enc_out


