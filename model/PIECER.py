# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertModel
from dgl.nn.pytorch import GraphConv, GATConv, GINConv
import dgl
import json
import pickle
from nltk.corpus import stopwords 
import nltk.stem.snowball as snowball
from string import punctuation
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# mask 为 1 的地方保持原状，mask 为 0 的地方设为负无穷 (-1e30)
def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # 原问题: do we need * mask after target? 回答: 影响不大，乘上更合理。


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            # nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')  # !!!!! BERT_finetune 版本如果用 init 会变成 nan，试试不用 init: 不是这里的问题
        else:
            self.relu = False
            # nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2) # batch, nwords, d_model;
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale) # batch, nwords, d_model;
    return (x + signal.to(x.get_device())).transpose(1, 2) # batch, d_model, nwords;


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):  # 此处是 transformer 的 pos embedding
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):
    def __init__(self, layer_num, size, dropout=0.1):
        super().__init__()
        self.n = layer_num
        # self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.nonlinear = nn.ModuleList([Initialized_Conv1d(size, size, relu=True, bias=True) for _ in range(self.n)])  # 原始代码这里没有加 relu，修正一下
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])
        self.dropout = dropout

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))  # [batch_size, hidden_size, length]
            nonlinear = self.nonlinear[i](x)  # [batch_size, hidden_size, length]
            nonlinear = F.dropout(nonlinear, p=self.dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x  # [batch_size, hidden_size, length]
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries  # batch, d_model, nwords
        memory = self.mem_conv(memory)  # batch, d_model * 2, nwords
        query = self.query_conv(queries)  # batch, d_model, nwords
        memory = memory.transpose(1, 2)  # batch, nwords, d_model * 2
        query = query.transpose(1, 2)  # batch, nwords, d_model
        Q = self.split_last_dim(query, self.num_head)  # batch, num_head, nwords, d_model / num_head
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(memory, self.d_model, dim=2)]
        # batch, num_head, nwords, d_model / num_head; batch, num_head, nwords, d_model / num_head

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head ** -0.5  # batch, num_head, nwords, d_model / num_head
        x = self.dot_product_attention(Q, K, V, mask = mask)  # batch, num_head, nwords, d_model / num_head
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)  # batch, d_model, nwords

    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        """
        dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0,1,3,2))  # batch, num_head, nwords, depth; batch, num_head, depth, nwords -> batch, num_head, nwords, nwords
        if bias:
            logits += self.bias
        if mask is not None:  # batch, nwords
            shapes = [x if x != None else -1 for x in list(logits.size())]  # value: [batch, num_head, nwords, nwords]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])  # batch, 1, 1, nwords
            logits = mask_logits(logits, mask)  # batch, num_head, nwords, nwords
        weights = F.softmax(logits, dim=-1)  # batch, num_head, nwords, nwords
        # dropping out the attention links for each of the heads，这里需要也 dropout 一部分 attention 的权重，跟 transformer 中的实现相同
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # batch, num_head, nwords, nwords
        return torch.matmul(weights, v)  # batch, num_head, nwords, nwords; batch, num_head, nwords, depth -> batch, num_head, nwords, depth

    def split_last_dim(self, x, n):
        """
        Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """
        Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.reshape(new_shape)
        return ret


class Embedding(nn.Module):
    def __init__(self, wemb_dim, cemb_dim, d_model,
                 dropout_w=0.1, dropout_c=0.05):
        super().__init__()
        self.conv2d = nn.Conv2d(cemb_dim, d_model, kernel_size = (1,5), padding=0, bias=True)
        # nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')  # !!!!! 试试不用这个初始化
        self.conv1d = Initialized_Conv1d(wemb_dim + d_model, d_model, bias=False)
        self.high = Highway(2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.permute(0, 3, 1, 2).contiguous()  # batch, cemb_dim, nwords, nchars
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)  # batch, cemb_dim, nwords, nchars'
        ch_emb = F.relu(ch_emb)
        # !!!!! 此处取 max 是否合理? 如果是这样的话，相当于 char embedding 部分就只能看到某个 5 字符的连续片段了
        ch_emb, _ = torch.max(ch_emb, dim=3)  # batch, d_model, nwords

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)  # batch, wemb_dim, nwords
        emb = torch.cat([ch_emb, wd_emb], dim=1)  # batch, d_model + wemb_dim, nwords
        emb = self.conv1d(emb)  # batch, d_model, nwords
        emb = self.high(emb)  # batch, d_model, nwords
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, d_model, num_head, kernel_size, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, kernel_size) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        # total_layers = (self.conv_num + 1) * blks
        total_layers = (self.conv_num + 2) * blks  # !!!!! 原始代码这里写成了 1，但应该是 2
        dropout = self.dropout
        out = PosEncoder(x)  # batch, d_model, nwords
        for i, conv in enumerate(self.convs):
            res = out  # batch, d_model, nwords
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)  # batch, d_model, nwords
            out = conv(out)  # batch, d_model, nwords
            # 此处的 dropout 取名有点问题，实际上是一个决定第 l 层是否 survival 的超参数，不过在数值上没问题
            out = self.layer_dropout(out, res, dropout * float(l) / total_layers)  # batch, d_model, nwords
            l += 1
        
        res = out  # batch, d_model, nwords
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        out = F.dropout(out, p=dropout, training=self.training)  # batch, d_model, nwords
        out = self.self_att(out, mask)  # batch, d_model, nwords
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)  # batch, d_model, nwords
        l += 1

        res = out  # batch, d_model, nwords
        out = self.norm_2(out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        out = F.dropout(out, p=dropout, training=self.training)  # batch, d_model, nwords
        out = self.FFN_1(out)  # batch, d_model, nwords
        out = self.FFN_2(out)  # batch, d_model, nwords
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)  # batch, d_model, nwords
        return out

    def layer_dropout(self, inputs, residual, dropout):
        # 在训练的时候可能会小概率地随机丢掉一些层，训练测试不一致？
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            pred = False  # 强制不要 layer_drop 了
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return F.dropout(inputs, dropout, training=self.training) + residual


class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.uniform_(w4C, -math.sqrt(1.0 / d_model), math.sqrt(1.0 / d_model))  # !!!!! 暂时换成默认的简单初始化，不使用 xavier 初始化
        nn.init.uniform_(w4Q, -math.sqrt(1.0 / d_model), math.sqrt(1.0 / d_model))
        nn.init.uniform_(w4mlu, -math.sqrt(1.0 / d_model), math.sqrt(1.0 / d_model))
        # nn.init.xavier_uniform_(w4C)
        # nn.init.xavier_uniform_(w4Q)
        # nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)  # batch, lc, d_model
        Q = Q.transpose(1, 2)  # batch, lq, d_model
        batch_size_c = C.size()[0]  # batch
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)  # batch, lc, lq
        Cmask = Cmask.view(batch_size_c, Lc, 1)  # batch, lc, 1
        Qmask = Qmask.view(batch_size_c, 1, Lq)  # batch, 1, lq
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)  # batch, lc, lq, 对 Q 维度做 softmax, C 中每个词对 Q 中每个词的注意力
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)  # batch, lc, lq, 对 C 维度做 softmax, Q 中每个词对 C 中每个词的注意力
        A = torch.bmm(S1, Q)  # batch, lc, d_model, C 中每个词对 Q 的注意力加权表示
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)  # batch, lc, d_model, C 中每个词对 C 的注意力加权表示，此处注意力是通过 C 到 Q 再到 C 得到的
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)  # batch, lc, d_model * 4, C 中每个词的不同表示的拼接
        return out.transpose(1, 2)  # batch, d_model * 4, lc

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)  # batch, lc, d_model
        Q = F.dropout(Q, p=dropout, training=self.training)  # batch, lq, d_model
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])  # batch, lc, lq
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])  # batch, lc, lq
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))  # batch, lc, d_model; batch, d_model, lq -> batch lc, lq; 
        res = subres0 + subres1 + subres2  # batch, lc, lq
        res += self.bias  # batch, lc, lq
        return res


class Pointer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model * 2 + 64, 1)
        self.w2 = Initialized_Conv1d(d_model * 2 + 64, 1)

    def forward(self, M1, M2, M3, mask, feat_ner):
        X1 = torch.cat([M1, M2, feat_ner], dim=1)  # batch, d_model * 2 + 64, lc
        X2 = torch.cat([M1, M3, feat_ner], dim=1)  # batch, d_model * 2 + 64, lc
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)  # batch, lc
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)  # batch, lc
        return Y1, Y2


class Pointer2(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model + 64, 1)
        self.w2 = Initialized_Conv1d(d_model + 64, 1)

    def forward(self, M, mask, feat_ner):
        X1 = torch.cat([M, feat_ner], dim=1)  # batch, d_model + 64, lc
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)  # batch, lc

        X2 = torch.cat([M, feat_ner], dim=1)  # batch, d_model + 64, lc
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)  # batch, lc

        return Y1, Y2


class SelfMatching(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super().__init__()
        # version other
        # self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        # self.FFN_1 = Initialized_Conv1d(d_model, d_model, bias=True)
        # self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        # self.norm_1 = nn.LayerNorm(d_model)
        # self.norm_2 = nn.LayerNorm(d_model)
        # self.dropout = dropout
        # version 5: original
        conv_num = 2
        kernel_size = 5
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, kernel_size) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask):
        # version 0: original simplified
        # out = self.norm_1(x.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        # out = self.self_att(out, mask)  # batch, d_model, nwords

        # out = self.norm_2(out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        # out = F.dropout(out, p=self.dropout, training=self.training)  # batch, d_model, nwords
        # out = self.FFN(out)  # batch, d_model, nwords

        # version 1: trm (wrong)
        # res = x
        # out = self.self_att(x, mask)  # batch, d_model, nwords
        # out = out + res
        # out = self.norm_1(out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords

        # res = out
        # out = self.FFN(out)  # batch, d_model, nwords
        # out = F.dropout(out, p=self.dropout, training=self.training)  # batch, d_model, nwords
        # out = out + res
        # out = self.norm_2(out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords

        # version 2: trm norm first (wrong)
        # x = self.norm_1(x.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        # res = x
        # out = self.self_att(x, mask)  # batch, d_model, nwords
        # out = out + res

        # out = self.norm_2(out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        # res = out
        # out = self.FFN(out)  # batch, d_model, nwords
        # out = F.dropout(out, p=self.dropout, training=self.training)  # batch, d_model, nwords
        # out = out + res

        # version 3: norm first, no FFN, + res
        # x = self.norm_1(x.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        # res = x
        # out = self.self_att(x, mask)  # batch, d_model, nwords
        # out = out + res

        # # version 4: trm (right)
        # # out = PosEncoder(x)  # (pos) batch, d_model, nwords
        # out = x

        # res = out  # (res) batch, d_model, nwords
        # out = self.self_att(x, mask)  # (attention) batch, d_model, nwords
        # out = res + F.dropout(out, p=self.dropout, training=self.training)  # (add) batch, d_model, nwords
        # out = self.norm_1(out.transpose(1,2)).transpose(1,2)  # (norm) batch, d_model, nwords

        # res = out  # (res) batch, d_model, nwords
        # out = self.FFN_1(out)  # (FFN_1) batch, d_model, nwords
        # out = F.relu(out)  # (FFN_relu)batch, d_model, nwords
        # out = self.FFN_2(out)  # (FFN_2) batch, d_model, nwords
        # out = res + F.dropout(out, p=self.dropout, training=self.training)  # (add) batch, d_model, nwords
        # out = self.norm_2(out.transpose(1,2)).transpose(1,2)  # (norm) batch, d_model, nwords

        # version 5: original
        out = PosEncoder(x)  # batch, d_model, nwords
        for i, conv in enumerate(self.convs):
            res = out  # batch, d_model, nwords
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)  # batch, d_model, nwords
            out = conv(out)  # batch, d_model, nwords
            out = F.dropout(out, p=self.dropout, training=self.training) + res  # batch, d_model, nwords
        
        res = out  # batch, d_model, nwords
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        out = F.dropout(out, p=self.dropout, training=self.training)  # batch, d_model, nwords
        out = self.self_att(out, mask)  # batch, d_model, nwords
        out = F.dropout(out, p=self.dropout, training=self.training) + res  # batch, d_model, nwords

        res = out  # batch, d_model, nwords
        out = self.norm_2(out.transpose(1,2)).transpose(1,2)  # batch, d_model, nwords
        out = F.dropout(out, p=self.dropout, training=self.training)  # batch, d_model, nwords
        out = self.FFN_1(out)  # batch, d_model, nwords
        out = self.FFN_2(out)  # batch, d_model, nwords
        out = F.dropout(out, p=self.dropout, training=self.training) + res  # batch, d_model, nwords
        
        return out


class PIECER(nn.Module):
    def __init__(self, word_mat, char_mat,
                 c_max_len, q_max_len, d_model, train_cemb=False, finetune_wemb=False, 
                 model='QANet', large=False, ptm_dir=None, 
                 use_kg_gcn=False, gcn_pos='both', gcn_num_layer=2, 
                 pad=0, use_ent_emb=False, after_matching=False, ev_tensor=None, ent2id=None, dropout=0.1, num_head=1): 
        super().__init__()
        self.PAD = pad
        self.Lc = c_max_len
        self.Lq = q_max_len
        self.dropout = dropout
        self.d_model = d_model
        self.model = model
        self.large = large
        self.ner_emb = nn.Embedding(5, 64)
        self.use_kg_gcn = use_kg_gcn
        self.gcn_pos = gcn_pos
        self.gcn_num_layer = gcn_num_layer
        self.num_head = num_head
        self.use_ent_emb = use_ent_emb
        self.after_matching = after_matching

        if self.model == 'QANet': 
            if train_cemb:
                self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False, padding_idx=0)  # 增加设置了 padding_idx
            else:
                self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=True, padding_idx=0)  # 增加设置了 padding_idx
            # !!!!! 此处冻结改为在 optimizer.step 之前置 0 部分梯度来手动实现。暂时还是不要这样了
            self.word_emb = nn.Embedding.from_pretrained(word_mat, freeze=(not finetune_wemb), padding_idx=0)  # 增加设置了 padding_idx
            wemb_dim = word_mat.shape[1]
            cemb_dim = char_mat.shape[1]
            self.emb = Embedding(wemb_dim, cemb_dim, d_model)
            self.emb_enc = EncoderBlock(conv_num=4, d_model=d_model, num_head=num_head, kernel_size=7, dropout=dropout)  # 此处的 dropout 改为了设置的参数，原来是直接设置为 0.1
            self.cq_att = CQAttention(d_model=d_model)
            self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
            self.model_enc_blks = nn.ModuleList(
                [EncoderBlock(conv_num=2, d_model=d_model, num_head=num_head, kernel_size=5, dropout=dropout) for _ in range(7)]
            )  # 此处的 dropout 改为了设置的参数，原来是直接设置为 0.1
            self.out = Pointer(d_model)

        if self.model == 'BERT':
            assert ptm_dir is not None
            from pytorch_transformers import BertTokenizer, BertModel
            self.bert_tokenizer = BertTokenizer.from_pretrained(ptm_dir)
            self.fast_tokenizer_dict = {}
            self.bert_model = BertModel.from_pretrained(ptm_dir)
            self.bt_high = Highway(2, d_model)
            if self.after_matching:
                self.model_enc_blk = SelfMatching(d_model=d_model, num_head=num_head, dropout=dropout)
                self.out_bert = Pointer2(d_model)
            else:
                self.out_bert = Pointer2(d_model)
        
        if self.model == 'RoBERTa': 
            assert ptm_dir is not None
            from pytorch_transformers import RobertaTokenizer, RobertaModel
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained(ptm_dir)
            self.fast_tokenizer_dict = {}
            self.roberta_model = RobertaModel.from_pretrained(ptm_dir)
            self.rbt_high = Highway(2, d_model)
            if self.after_matching:
                self.model_enc_blk = SelfMatching(d_model=d_model, num_head=num_head, dropout=dropout)
                self.out_roberta = Pointer2(d_model)
            else:
                self.out_roberta = Pointer2(d_model)

        # 直接注入知识
        if self.use_ent_emb:
            self.ent_emb = nn.Embedding.from_pretrained(ev_tensor, freeze=True)  # 增加设置了 padding_idx
            self.ent2id = ent2id
            # 结合方式 1: ScalarGating
            self.ent_resizer = nn.Linear(100, self.d_model)
            self.inj_gate = nn.Linear(self.d_model * 2, 1)
            # !!!!!!! 结合方式 2: LinearCombiner
            # self.ent_combiner = nn.Linear(self.d_model + 100, self.d_model)
            # !!!!!!! 结合方式 3: TensorGating
            # self.ent_resizer = nn.Linear(100, self.d_model)
            # self.inj_gate = nn.Linear(self.d_model * 2, self.d_model)

        # 知识引导交互
        if self.use_kg_gcn: 
            # used for graph preparation 
            self.snowball_stemmer = snowball.SnowballStemmer("english")
            self.stop_words = set(stopwords.words('english'))
            self.punct = punctuation
            
            # normal
            if not self.load_tools():
                self.fast_stemmer = {}
                self.relations, self.id2rel, self.rel2id = self.build_relations()
                self.fast_graph = {}

            # test graph
            # self.fast_stemmer = {}
            # self.relations, self.id2rel, self.rel2id = self.build_relations()
            # self.fast_graph = {}

            # GAT 原文是 8 头，但这里的图可能会很稀疏，改为 4 头
            if self.gcn_pos == 'emb' or self.gcn_pos == 'both': 
                # gin_linear_list = nn.ModuleList(
                #     nn.Linear(d_model, d_model) for _ in range(self.gcn_num_layer)
                # )
                self.gcn_list = nn.ModuleList(
                    GATConv(d_model, d_model, num_heads=4) for _ in range(self.gcn_num_layer)
                    # GraphConv(d_model, d_model) for _ in range(self.gcn_num_layer)
                    # GINConv(gin_linear_list[i], 'max') for i in range(self.gcn_num_layer)
                )
                self.gate_list = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(self.gcn_num_layer))
            if self.gcn_pos == 'enc' or self.gcn_pos == 'both': 
                # gin_linear_list2 = nn.ModuleList(
                #     nn.Linear(d_model, d_model) for _ in range(self.gcn_num_layer)
                # )
                self.gcn_list2 = nn.ModuleList(
                    GATConv(d_model, d_model, num_heads=4) for _ in range(self.gcn_num_layer)
                    # GraphConv(d_model, d_model) for _ in range(self.gcn_num_layer)
                    # GINConv(gin_linear_list2[i], 'max') for i in range(self.gcn_num_layer)
                )
                self.gate_list2 = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(self.gcn_num_layer))
    
    def check_time(self, st_time, print_info):
        print('==================== ' + print_info + ' ====================')
        print(time.time() - st_time)
        return time.time()
    
    def fast_tokenizer(self, word, tokenizer):
        if word in self.fast_tokenizer_dict:
            return self.fast_tokenizer_dict[word]
        ids = tokenizer.encode(word)
        self.fast_tokenizer_dict[word] = ids
        return ids

    def forward(self, Cwid, Ccid, Qwid, Qcid, context_tokens=None, ques_tokens=None, Cnid=None):
        # padding mask
        maskC = (torch.ones_like(Cwid) * self.PAD != Cwid).float()  # batch, lc
        maskQ = (torch.ones_like(Qwid) * self.PAD != Qwid).float()  # batch, lq
        # cur_time = time.time()
        # cur_time = self.check_time(cur_time, 'begin')
        # QANet embedding
        if self.model == 'QANet': 
            Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)  # batch, lc, wemb_dim; batch, lc, nchars, cemb_dim
            Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)  # batch, lq, wemb_dim; batch, lq, nchars, cemb_dim
            C, Q = self.emb(Cc, Cw, self.Lc), self.emb(Qc, Qw, self.Lq)  # batch, d_model, lc; batch, d_model, lq

        # BERT embedding
        if self.model == 'BERT':
            qc_tokens = []
            for i in range(len(context_tokens)):
                qc_tokens.append(ques_tokens[i] + context_tokens[i])
            QC = self.bert_embedding(qc_tokens)  # batch, d_model, lq + lc
            Q, C = torch.split(QC, [self.Lq, self.Lc], dim=2)
        
        # RoBERTa embedding
        if self.model == 'RoBERTa': 
            qc_tokens = []
            for i in range(len(context_tokens)):
                qc_tokens.append(ques_tokens[i] + context_tokens[i])
            QC = self.roberta_embedding(qc_tokens)  # batch, d_model, lq + lc
            Q, C = torch.split(QC, [self.Lq, self.Lc], dim=2)
            # cur_time = self.check_time(cur_time, 'RoBERTa encoding')

        # 建图
        if self.use_kg_gcn:
            cq_graphs = []
            for i in range(len(context_tokens)): 
                cq_tokens = context_tokens[i] + ques_tokens[i]
                cq_graphs.append(self.build_sentence_graph(cq_tokens))
            cq_graph_batch = dgl.batch(cq_graphs)
        # cur_time = self.check_time(cur_time, 'graph construction')
        
        # 直接注入知识
        if self.use_ent_emb:
            C = self.KnInjection(C, context_tokens)
            Q = self.KnInjection(Q, ques_tokens)
        # cur_time = self.check_time(cur_time, 'injection')
        
        # pos-emb: C 和 Q 中和知识有关的词互相进行交互
        if self.use_kg_gcn and (self.gcn_pos == 'emb' or self.gcn_pos == 'both'): 
            C, Q = self.KnGuidedHighwayGAT(C, Q, cq_graph_batch, self.gate_list, self.gcn_list)
        # cur_time = self.check_time(cur_time, 'GAT interaction')

        # QANet model and decode
        if self.model == 'QANet': 
            C = self.emb_enc(C, maskC, 1, 1)  # batch, d_model, lc; 
            Q = self.emb_enc(Q, maskQ, 1, 1)  # batch, d_model, lq; 
            # pos-enc: C 和 Q 中和知识有关的词互相进行交互
            if self.use_kg_gcn and (self.gcn_pos == 'enc' or self.gcn_pos == 'both'): 
                C, Q = self.KnGuidedHighwayGAT(C, Q, cq_graph_batch, self.gate_list2, self.gcn_list2)
            X = self.cq_att(C, Q, maskC, maskQ)  # batch, d_model * 4, lc
            M0 = self.cq_resizer(X)  # batch, d_model, lc
            M0 = F.dropout(M0, p=self.dropout, training=self.training)  # batch, d_model, lc
            for i, blk in enumerate(self.model_enc_blks):
                M0 = blk(M0, maskC, i*(2+2)+1, 7)  # batch, d_model, lc
            M1 = M0  # batch, d_model, lc
            for i, blk in enumerate(self.model_enc_blks):
                M0 = blk(M0, maskC, i*(2+2)+1, 7)  # batch, d_model, lc
            M2 = M0  # batch, d_model, lc
            M0 = F.dropout(M0, p=self.dropout, training=self.training)  # batch, d_model, lc
            for i, blk in enumerate(self.model_enc_blks):
                M0 = blk(M0, maskC, i*(2+2)+1, 7)  # batch, d_model, lc
            M3 = M0  # batch, d_model, lc
            feat_ner = self.ner_emb(Cnid).transpose(1, 2)  # batch, 64, lc; 增加 ner 的特征以帮助学习答案位置
            p1, p2 = self.out(M1, M2, M3, maskC, feat_ner)  # batch, lc; batch, lc
            return p1, p2
        
        # BERT model and decode
        if self.model == 'BERT': 
            if self.after_matching:
                M0 = self.model_enc_blk(C, maskC)  # batch, d_model, lc
                feat_ner = self.ner_emb(Cnid).transpose(1, 2)  # batch, 64, lc; 增加 ner 的特征以帮助学习答案位置
                p1, p2 = self.out_bert(M0, maskC, feat_ner)  # batch, lc; batch, lc
                # M0 = C
                # M0 = self.model_enc_blk(M0, maskC)  # batch, d_model, lc
                # M1 = M0  # batch, d_model, lc
                # M0 = self.model_enc_blk(M0, maskC)  # batch, d_model, lc
                # M2 = M0  # batch, d_model, lc
                # M0 = self.model_enc_blk(M0, maskC)  # batch, d_model, lc
                # M3 = M0  # batch, d_model, lc
                # feat_ner = self.ner_emb(Cnid).transpose(1, 2)  # batch, 64, lc; 增加 ner 的特征以帮助学习答案位置
                # p1, p2 = self.out_bert(M1, M2, M3, maskC, feat_ner)  # batch, lc; batch, lc
            else:
                feat_ner = self.ner_emb(Cnid).transpose(1, 2)  # batch, 64, lc; 增加 ner 的特征以帮助学习答案位置
                p1, p2 = self.out_bert(C, maskC, feat_ner)  # batch, lc; batch, lc
            return p1, p2

        # RoBERTa model and decode
        if self.model == 'RoBERTa': 
            if self.after_matching:
                M0 = self.model_enc_blk(C, maskC)  # batch, d_model, lc
                feat_ner = self.ner_emb(Cnid).transpose(1, 2)  # batch, 64, lc; 增加 ner 的特征以帮助学习答案位置
                p1, p2 = self.out_roberta(M0, maskC, feat_ner)  # batch, lc; batch, lc
                # M0 = C
                # M0 = self.model_enc_blk(M0, maskC)  # batch, d_model, lc
                # M1 = M0  # batch, d_model, lc
                # M0 = self.model_enc_blk(M0, maskC)  # batch, d_model, lc
                # M2 = M0  # batch, d_model, lc
                # M0 = self.model_enc_blk(M0, maskC)  # batch, d_model, lc
                # M3 = M0  # batch, d_model, lc
                # feat_ner = self.ner_emb(Cnid).transpose(1, 2)  # batch, 64, lc; 增加 ner 的特征以帮助学习答案位置
                # p1, p2 = self.out_roberta(M1, M2, M3, maskC, feat_ner)  # batch, lc; batch, lc
                # cur_time = self.check_time(cur_time, 'matching & pointing')
            else:
                feat_ner = self.ner_emb(Cnid).transpose(1, 2)  # batch, 64, lc; 增加 ner 的特征以帮助学习答案位置
                p1, p2 = self.out_roberta(C, maskC, feat_ner)  # batch, lc; batch, lc
                # cur_time = self.check_time(cur_time, 'pointing')
            return p1, p2
    
    def KnInjection(self, tensors, tokens): 
        num_batchs = tensors.shape[0]
        l_tokens = tensors.shape[2]
        
        ent_id = torch.zeros((num_batchs, l_tokens)).long().cuda()  # batch, len
        ent_mask = torch.zeros((num_batchs, l_tokens)).float().cuda()  # batch, len

        for batch_idx in range(num_batchs): 
            for token_idx in range(l_tokens): 
                cur_word = tokens[batch_idx][token_idx]
                if cur_word == '<PAD>' or cur_word == '[PAD]' or cur_word == '<pad>': 
                    continue
                cur_word = cur_word.lower()
                if cur_word in self.stop_words or cur_word in self.punct:
                    continue
                stem_word = self.fast_stem(cur_word)
                if stem_word not in self.relations or stem_word not in self.ent2id:
                    continue
                ent_id[batch_idx, token_idx] = self.ent2id[stem_word]
                ent_mask[batch_idx, token_idx] = 1.0
        
        # 结合方式 1: ScalarGating
        ent_emb = self.ent_emb(ent_id)  # batch, len, 100
        ent_emb = self.ent_resizer(ent_emb)  # batch, len, d_model

        # normal
        inj_gate = torch.sigmoid(self.inj_gate(torch.cat((tensors.transpose(1, 2), ent_emb), dim=2)))  # batch, len, 1
        inj_gate = inj_gate * ent_mask.unsqueeze(2)  # batch, len, 1
        new_tensors = tensors.transpose(1, 2) * (1 - inj_gate) + ent_emb * inj_gate
        new_tensors = new_tensors.transpose(1, 2)
        
        # test w/o gating
        # new_tensors = tensors.transpose(1, 2) + ent_emb
        # new_tensors = new_tensors.transpose(1, 2)
        
        # test half gating
        # inj_gate = 0.5
        # new_tensors = tensors.transpose(1, 2) * (1 - inj_gate) + ent_emb * inj_gate
        # new_tensors = new_tensors.transpose(1, 2)

        # !!!!!!! 结合方式 2: LinearCombiner
        # ent_emb = self.ent_emb(ent_id) * ent_mask.unsqueeze(2) # batch, len, 100
        # new_tensors = self.ent_combiner(torch.cat((tensors.transpose(1, 2), ent_emb), dim=2))  # batch, len, d_model
        # new_tensors = new_tensors.transpose(1, 2)  # batch, d_model, len
        # new_tensors = F.relu(F.dropout(new_tensors, p=self.dropout, training=self.training))  # batch, d_model, len

        # !!!!!!! 结合方式 3: TensorGating
        # ent_emb = self.ent_emb(ent_id)  # batch, len, 100
        # ent_emb = self.ent_resizer(ent_emb)  # batch, len, d_model

        # inj_gate = torch.sigmoid(self.inj_gate(torch.cat((tensors.transpose(1, 2), ent_emb), dim=2)))  # batch, len, d_model
        # inj_gate = inj_gate * ent_mask.unsqueeze(2)  # batch, len, 1
        # new_tensors = tensors.transpose(1, 2) * (1 - inj_gate) + ent_emb * inj_gate
        # new_tensors = new_tensors.transpose(1, 2)
        
        return new_tensors

    def KnGuidedHighwayGAT(self, C, Q, cq_graph_batch, gate_list, gcn_list): 
        # C: batch, d_model, lc; Q: batch, d_model, lq
        CQ = torch.cat((C, Q), dim=2)  # batch, d_model, lc + lq
        CQ = CQ.transpose(1, 2)  # batch, lc + lq, d_model
        CQ = CQ.reshape(-1, self.d_model)  # batch * (lc + lq), d_model
        for i in range(self.gcn_num_layer):
            gate_CQ = torch.sigmoid(gate_list[i](CQ))
            gcn_CQ = gcn_list[i](cq_graph_batch, CQ)
            gcn_CQ = gcn_CQ.mean(dim=1)
            gcn_CQ = F.relu(F.dropout(gcn_CQ, p=self.dropout, training=self.training))
            CQ = gate_CQ * gcn_CQ + (1 - gate_CQ) * CQ
        CQ = CQ.view(C.shape[0], -1, self.d_model)  # batch, lc + lq, d_model
        CQ = CQ.transpose(1, 2)  # batch, d_model, lc + lq
        C, Q = torch.split(CQ, [self.Lc, self.Lq], dim=2)
        return C, Q
        
    def load_tools(self):
        if not os.path.exists('./data/processed/ReCoRD/tool_info.pkl'): 
            return False
        with open('./data/processed/ReCoRD/tool_info.pkl', 'rb') as f:
            tool_info = pickle.load(f)
        self.fast_stemmer = tool_info['fast_stemmer']
        self.fast_graph = tool_info['fast_graph']
        self.relations = tool_info['relations']
        self.id2rel = tool_info['id2rel']
        self.rel2id = tool_info['rel2id']
        return True
    
    def save_tools(self):
        tool_info = {
            'fast_stemmer': self.fast_stemmer, 
            'fast_graph': self.fast_graph, 
            'relations': self.relations, 
            'id2rel': self.id2rel, 
            'rel2id': self.rel2id, 
        }
        with open('./data/processed/ReCoRD/tool_info.pkl', 'wb') as f:
            pickle.dump(tool_info, f)

    def bert_embedding(self, tokens):
        for i, sentence_tokens in enumerate(tokens):
            for j in range(len(sentence_tokens)):
                if tokens[i][j] == '<PAD>' or tokens[i][j] == '<pad>':
                    tokens[i][j] = '[PAD]'
                if tokens[i][j] == '@placeholder': 
                    tokens[i][j] = '[MASK]'
                
        input_ids_list = []
        input_segs_list = []
        input_masks_list = []
        piece_ranges_list = []
        maxlen = -1
        for i, sentence_tokens in enumerate(tokens):
            input_ids = []
            input_segs = []
            input_masks = []
            piece_ranges = []
            now_seg = 0
            input_ids += self.fast_tokenizer('[CLS]', self.bert_tokenizer)  # 此处试试添加特殊token [CLS] 和 [SEP] 是否影响性能：会更好
            input_segs += [now_seg]
            input_masks += [1]
            for j, word in enumerate(sentence_tokens):
                if j > 0 and sentence_tokens[j] != '[PAD]' and sentence_tokens[j - 1] == '[PAD]':  # 此处试试添加特殊token [CLS] 和 [SEP] 是否影响性能：会更好
                    input_ids += self.fast_tokenizer('[SEP]', self.bert_tokenizer)
                    input_segs += [now_seg]
                    input_masks += [1]
                    if now_seg == 0:
                        now_seg = 1
                if word == '[PAD]': 
                    piece_ranges.append((-1, -1))
                    continue
                elif len(word.split()) == 0:
                    word_ids = [0]
                else:
                    word_ids = self.fast_tokenizer(word, self.bert_tokenizer)
                if len(word_ids) == 0:
                    word_ids = [0]
                range_start = len(input_ids)
                input_ids += word_ids
                input_segs += [now_seg] * len(word_ids)
                if len(word_ids) == 1 and word_ids[0] == 0: 
                    input_masks += [0]
                else:
                    input_masks += [1] * len(word_ids)
                range_end = len(input_ids)
                piece_ranges.append((range_start, range_end))
            input_ids += self.fast_tokenizer('[SEP]', self.bert_tokenizer)
            input_segs += [now_seg]
            input_masks += [1]
            input_ids_list.append(input_ids)
            input_segs_list.append(input_segs)
            input_masks_list.append(input_masks)
            piece_ranges_list.append(piece_ranges)
            maxlen = max(maxlen, len(input_ids))
        for i, input_ids in enumerate(input_ids_list):
            if len(input_ids) < maxlen:
                input_ids_list[i] = input_ids_list[i] + [0 for i in range(maxlen - len(input_ids))]
                input_segs_list[i] = input_segs_list[i] + [1 for i in range(maxlen - len(input_ids))]
                input_masks_list[i] = input_masks_list[i] + [0 for i in range(maxlen - len(input_ids))]
        input_ids = torch.tensor(input_ids_list).cuda()  # batch, maxlen
        input_segs = torch.tensor(input_segs_list).long().cuda()  # batch, maxlen
        input_masks = torch.tensor(input_masks_list).float().cuda()  # batch, maxlen

        # if input_ids.shape[1] <= 512:
        #     outputs = self.bert_model(input_ids, token_type_ids=input_segs, attention_mask=input_masks)[0]  # batch, maxlen, d_bert
        # else:
        #     input_ids1 = input_ids[:, :512]  # batch, 512
        #     input_segs1 = input_segs[:, :512]  # batch, 512
        #     input_masks1 = input_masks[:, :512]  # batch, 512
        #     input_ids2 = input_ids[:, 512:]  # batch, maxlen - 512
        #     input_segs2 = input_segs[:, 512:] - input_segs[:, 512:]  # batch, maxlen - 512, 全 0
        #     input_masks2 = input_masks[:, 512:]  # batch, maxlen - 512
        #     outputs1 = self.bert_model(input_ids1, token_type_ids=input_segs1, attention_mask=input_masks1)[0]  # batch, 512, d_bert
        #     outputs2 = self.bert_model(input_ids2, token_type_ids=input_segs2, attention_mask=input_masks2)[0]  # batch, maxlen - 512, d_bert
        #     outputs = torch.cat((outputs1, outputs2), dim=1)  # batch, maxlen, d_bert
        # print('bert model:', time.time() - start_time)
        # for i, sentence_tokens in enumerate(tokens):
        #     for j, word in enumerate(sentence_tokens):
        #         word_embeddings[i, j, :] = torch.mean(outputs[i, piece_ranges_list[i][j][0]:piece_ranges_list[i][j][1], :], dim=0)  # batch, nwords, d_bert
        # !!!!!!! 新逻辑，如果超长就直接截取
        if input_ids.shape[1] <= 512:
            outputs = self.bert_model(input_ids, token_type_ids=input_segs, attention_mask=input_masks)[0]  # batch, maxlen, d_bert
        else:
            input_ids1 = input_ids[:, :512]  # batch, 512
            input_segs1 = input_segs[:, :512]  # batch, 512
            input_masks1 = input_masks[:, :512]  # batch, 512
            outputs = self.bert_model(input_ids1, token_type_ids=input_segs1, attention_mask=input_masks1)[0]  # batch, 512, d_bert

        welist = []
        for i, sentence_tokens in enumerate(tokens):
            welist.append([])
            outputs_i = outputs[i]
            for j, word in enumerate(sentence_tokens):
                if word == '[PAD]': 
                    welist[-1].append(torch.zeros(outputs.shape[2]).cuda())
                elif piece_ranges_list[i][j][1] <= 512:
                    if piece_ranges_list[i][j][0] + 1 < piece_ranges_list[i][j][1]:
                        welist[-1].append(torch.mean(outputs_i[piece_ranges_list[i][j][0]:piece_ranges_list[i][j][1], :], dim=0))  # batch, nwords, d_bert
                    else:
                        welist[-1].append(outputs_i[piece_ranges_list[i][j][0]])  # batch, nwords, d_bert
                else:
                    welist[-1].append(torch.zeros(outputs.shape[2]).cuda())  # batch, nwords, d_bert
            welist[-1] = torch.stack(welist[-1], dim=0)
        word_embeddings = torch.stack(welist, dim=0)  # batch, nwords, d_bert

        word_embeddings = word_embeddings.transpose(1, 2)  # batch, d_bert, nwords
        word_embeddings = self.bt_high(word_embeddings)  # batch, d_model, nwords

        return word_embeddings
    
    def roberta_embedding(self, tokens):
        for i, sentence_tokens in enumerate(tokens):
            for j in range(len(sentence_tokens)):
                if tokens[i][j] == '<PAD>' or tokens[i][j] == '[PAD]':
                    tokens[i][j] = '<pad>'
                if tokens[i][j] == '@placeholder': 
                    tokens[i][j] = '<mask>'
        input_ids_list = []
        input_masks_list = []
        piece_ranges_list = []
        maxlen = -1
        for i, sentence_tokens in enumerate(tokens):
            input_ids = []
            input_masks = []
            piece_ranges = []
            input_ids += self.fast_tokenizer('<s>', self.roberta_tokenizer)  # special token for RoBERTa 
            input_masks += [1]
            for j, word in enumerate(sentence_tokens):
                if j > 0 and sentence_tokens[j] == '<pad>' and sentence_tokens[j - 1] != '<pad>':  # special token for RoBERTa
                    input_ids += self.fast_tokenizer('</s>', self.roberta_tokenizer)
                    input_masks += [1]
                if j > 0 and sentence_tokens[j] != '<pad>' and sentence_tokens[j - 1] == '<pad>':  # special token for RoBERTa
                    input_ids += self.fast_tokenizer('<s>', self.roberta_tokenizer)
                    input_masks += [1]
                if word == '<pad>': 
                    piece_ranges.append((-1, -1))
                    continue
                elif len(word.split()) == 0:
                    word_ids = [1]
                else:
                    word_ids = self.fast_tokenizer(word, self.roberta_tokenizer)
                if len(word_ids) == 0:
                    word_ids = [1]
                range_start = len(input_ids)
                input_ids += word_ids
                if len(word_ids) == 1 and word_ids[0] == 1: 
                    input_masks += [0]
                else:
                    input_masks += [1] * len(word_ids)
                range_end = len(input_ids)
                piece_ranges.append((range_start, range_end))
            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)
            piece_ranges_list.append(piece_ranges)
            maxlen = max(maxlen, len(input_ids))
        for i, input_ids in enumerate(input_ids_list):
            if len(input_ids) < maxlen:
                input_ids_list[i] = input_ids_list[i] + [1 for i in range(maxlen - len(input_ids))]
                input_masks_list[i] = input_masks_list[i] + [0 for i in range(maxlen - len(input_ids))]
        input_ids = torch.tensor(input_ids_list).cuda()  # batch, maxlen
        input_masks = torch.tensor(input_masks_list).float().cuda()  # batch, maxlen
        # if input_ids.shape[1] <= 512:
        #     outputs = self.roberta_model(input_ids, attention_mask=input_masks)[0]  # batch, maxlen, d_bert
        # else:
        #     input_ids1 = input_ids[:, :512]  # batch, 512
        #     input_masks1 = input_masks[:, :512]  # batch, 512
        #     input_ids2 = input_ids[:, 512:]  # batch, maxlen - 512
        #     input_masks2 = input_masks[:, 512:]  # batch, maxlen - 512
        #     outputs1 = self.roberta_model(input_ids1, attention_mask=input_masks1)[0]  # batch, 512, d_bert
        #     outputs2 = self.roberta_model(input_ids2, attention_mask=input_masks2)[0]  # batch, maxlen - 512, d_bert
        #     outputs = torch.cat((outputs1, outputs2), dim=1)  # batch, maxlen, d_bert
        # for i, sentence_tokens in enumerate(tokens):
        #     for j, word in enumerate(sentence_tokens):
        #         word_embeddings[i, j, :] = torch.mean(outputs[i, piece_ranges_list[i][j][0]:piece_ranges_list[i][j][1], :], dim=0)  # batch, nwords, d_bert
        # for i, sentence_tokens in enumerate(tokens):
        #     for j, word in enumerate(sentence_tokens):
        #         if piece_ranges_list[i][j][1] <= 512:
        #             word_embeddings[i, j, :] = torch.mean(outputs[i, piece_ranges_list[i][j][0]:piece_ranges_list[i][j][1], :], dim=0)  # batch, nwords, d_bert
        # !!!!!!! 新逻辑，如果超长就直接截取
        if input_ids.shape[1] <= 512:
            outputs = self.roberta_model(input_ids, attention_mask=input_masks)[0]  # batch, maxlen, d_bert
        else:
            input_ids1 = input_ids[:, :512]  # batch, 512
            input_masks1 = input_masks[:, :512]  # batch, 512
            outputs = self.roberta_model(input_ids1, attention_mask=input_masks1)[0]  # batch, 512, d_bert
        
        welist = []
        for i, sentence_tokens in enumerate(tokens):
            welist.append([])
            outputs_i = outputs[i]
            for j, word in enumerate(sentence_tokens):
                if word == '<pad>': 
                    welist[-1].append(torch.zeros(outputs.shape[2]).cuda())
                elif piece_ranges_list[i][j][1] <= 512:
                    if piece_ranges_list[i][j][0] + 1 < piece_ranges_list[i][j][1]:
                        welist[-1].append(torch.mean(outputs_i[piece_ranges_list[i][j][0]:piece_ranges_list[i][j][1], :], dim=0))  # batch, nwords, d_bert
                    else:
                        welist[-1].append(outputs_i[piece_ranges_list[i][j][0]])
                else:
                    welist[-1].append(torch.zeros(outputs.shape[2]).cuda())
            welist[-1] = torch.stack(welist[-1], dim=0)
        word_embeddings = torch.stack(welist, dim=0)  # batch, nwords, d_bert

        word_embeddings = word_embeddings.transpose(1, 2)  # batch, d_bert, nwords
        word_embeddings = self.rbt_high(word_embeddings)  # batch, d_model, nwords

        return word_embeddings

    def fast_stem(self, word):
        if '@#@' in word:
            w1, w2 = word.split('@#@')
            if w1 not in self.fast_stemmer:
                self.fast_stemmer[w1] = self.snowball_stemmer.stem(w1)
            if w2 not in self.fast_stemmer:
                self.fast_stemmer[w2] = self.snowball_stemmer.stem(w2)
            return self.fast_stemmer[w1] + '@#@' + self.fast_stemmer[w2]
        else:
            if word not in self.fast_stemmer:
                self.fast_stemmer[word] = self.snowball_stemmer.stem(word)
            return self.fast_stemmer[word]

    def build_relations(self, ):
        with open('./data/original/ConceptNet/simplified_triples_of_ConceptNet.json', 'r', encoding='utf-8') as f:
            ori_triplets = json.load(f)
        triplets = []
        for triplet in ori_triplets: 
            head_words = triplet[0].split('_')
            tail_words = triplet[2].split('_')
            if len(head_words) <= 2 and len(tail_words) <= 2: 
                triplet[0] = '@#@'.join(head_words)
                triplet[2] = '@#@'.join(tail_words)
                triplets.append(triplet)

        id2rel = ['self_loop', 'adj_word', 'same_word']
        considered_rel = ['/r/RelatedTo', '/r/FormOf', '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor', '/r/CapableOf', '/r/AtLocation', '/r/Causes', '/r/HasSubevent', '/r/HasPrerequisite', '/r/HasProperty', '/r/CreatedBy', '/r/Synonym', '/r/SymbolOf', '/r/DefinedAs', '/r/MannerOf', '/r/LocatedNear', '/r/SimilarTo', '/r/MadeOf', '/r/ReceivesAction']
        relations = {}
        for triplet in triplets:
            h, r, t = triplet
            if r not in considered_rel:
                continue
            if r not in id2rel:
                id2rel.append(r)
            if r + '(inverse)' not in id2rel:
                id2rel.append(r + '(inverse)')
            h = h.lower()
            t = t.lower()
            h = self.fast_stem(h)
            t = self.fast_stem(t)
            if h not in relations:
                relations[h] = {}
            if t not in relations[h]:
                relations[h][t] = set()
            relations[h][t].update([r])
            if t not in relations:
                relations[t] = {}
            if h not in relations[t]:
                relations[t][h] = set()
            relations[t][h].update([r + '(inverse)'])
        rel2id = {}
        for i, rel in enumerate(id2rel):
            rel2id[rel] = i
        return relations, id2rel, rel2id
    
    def build_sentence_graph(self, sentence):
        # homogeneous graph

        str_sent = '|'.join(sentence)
        if str_sent in self.fast_graph:
            return self.fast_graph[str_sent].to('cuda')

        u = []
        v = []

        # stemed same & knowledge-based edge
        for i, word1 in enumerate(sentence):
            if word1 == '<PAD>' or word1 == '[PAD]' or word1 == '<pad>': 
                continue
            for j, word2 in enumerate(sentence): 
                if word2 == '<PAD>' or word2 == '[PAD]' or word2 == '<pad>': 
                    continue
                if i == j:
                    continue
                word1 = word1.lower()
                word2 = word2.lower()
                if word1 in self.stop_words or word2 in self.stop_words:
                    continue
                if word1 in self.punct or word2 in self.punct:
                    continue
                # same word
                stem_w1 = self.fast_stem(word1)
                stem_w2 = self.fast_stem(word2)
                if stem_w1 == stem_w2: 
                    u.append(i)
                    v.append(j)
                    continue
                # word1 -> word2 or word2*
                if stem_w1 in self.relations:
                    if stem_w2 in self.relations[stem_w1]:
                        u.append(i)
                        v.append(j)
                        continue
                    if j < len(sentence) - 1:
                        stem_word2x = self.fast_stem(word2 + '@#@' + sentence[j + 1])
                        if stem_word2x in self.relations[stem_w1]: 
                            u.append(i)
                            v.append(j)
                            continue
                # word1* -> word2 or word2*
                if i < len(sentence) - 1:
                    stem_word1x = self.fast_stem(word1 + '@#@' + sentence[i + 1])
                    if stem_word1x in self.relations:
                        if stem_w2 in self.relations[stem_word1x]:
                            u.append(i)
                            v.append(j)
                            continue
                        if j < len(sentence) - 1:
                            stem_word2x = self.fast_stem(word2 + '@#@' + sentence[j + 1])
                            if stem_word2x in self.relations[stem_word1x]: 
                                u.append(i)
                                v.append(j)
                                continue
        
        # adjacent word edge
        # for i in range(len(sentence) - 1): 
        #     if sentence[i + 1] == '[PAD]' or sentence[i + 1] == '<PAD>' or sentence[i + 1] == '<pad>': 
        #         continue
        #     if sentence[i] == '[PAD]' or sentence[i] == '<PAD>' or sentence[i] == '<pad>': 
        #         continue
        #     u.append(i)
        #     v.append(i + 1)
        #     u.append(i + 1)
        #     v.append(i)

        u = torch.tensor(u).long().cuda()
        v = torch.tensor(v).long().cuda()
        G = dgl.graph((u, v), num_nodes=len(sentence))

        # self-loop edge
        G = dgl.add_self_loop(G)
        
        self.fast_graph[str_sent] = G.to('cpu')
        
        return G

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)