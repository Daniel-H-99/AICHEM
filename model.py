import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.src_embedding = nn.Embedding(args.vocab_size, args.d_e)
        self.pos_embedding = PositionEmbedding(args)
        self.encoder = Encoder(args)
        self.classifier = nn.Sequential(
            nn.Linear(args.d_e, args.d_h),
            nn.ReLU(),
            nn.Linear(args.d_h, args.d_h),
            nn.ReLU(),
            nn.Linear(args.d_h, 2)
        )
        self.input_refiner = nn.Conv1d(args.d_e, args.d_e, 3, stride=1, padding=1)
        self.projector = nn.Sequential(
            nn.Linear(args.d_e, args.d_e),
            nn.ReLU(),
            nn.Linear(args.d_e, args.d_e),
        )

    def forward(self, source, length):
        # source: (batch x source_length), length: (batch)
        # generate attention masks
        batch, src_len = source.shape[0], source.shape[1]
        enc_attn_mask = source.eq(self.args.pad_idx).unsqueeze(1).expand(-1, src_len, -1)   # (batch x source_length x source_length)
        # embedding
        src_embedding = self.src_embedding(source).transpose(1, 2)  # (batch x d_e x source_length)
        src_embedding = self.input_refiner(src_embedding)   # (batch x source_length x d_e)
        src_embedding = src_embedding.transpose(1, 2)
        encoder_i = self.pos_embedding(src_embedding)   # (batch x source_length x d_e)
        
        output = self.encoder(encoder_i, enc_attn_mask)                                  # (batch x source_length x d_e)
        # print('output: {}'.format(output.shape))
        output = output[torch.arange(batch).cuda(), length - 1]
        # print('output: {}'.format(output.shape))
        output = self.classifier(output)                         # (batch x 2) 
        return output
    
    def project(self, source, length):
        # source: (batch x source_length), length: (batch)
        # generate attention masks
        batch, src_len = source.shape[0], source.shape[1]
        enc_attn_mask = source.eq(self.args.pad_idx).unsqueeze(1).expand(-1, src_len, -1)   # (batch x source_length x source_length)
        # embedding
        src_embedding = self.src_embedding(source).transpose(1, 2)  # (batch x d_e x source_length)
        src_embedding = self.input_refiner(src_embedding).transpose(1, 2)   # (batch x source_length x d_e)
        encoder_i = self.pos_embedding(src_embedding)   # (batch x source_length x d_e)
        
        output = self.encoder(encoder_i, enc_attn_mask)                                  # (batch x source_length x d_e)
        # print('output: {}'.format(output.shape))
        output = output[torch.arange(batch).cuda(), length - 1]
        # print('output: {}'.format(output.shape))
        output = self.projector(output)                         # (batch x d_e) 
        return output

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.num_enc_layers)])
    def forward(self, input, attn_mask):
        # input: (batch x source_length x d_e)
        output = input
        for layer in self.layers:
            output = layer(output, attn_mask)   # (batch x source_length x d_e)
        return output
    
    
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.multiheadattn = MultiHeadAttn(args)
        self.posifeedforward = PosiFeedForward(args)
    def forward(self, input, attn_mask):
        # input: (batch x source_length x d_e)
        output = self.multiheadattn(input, input, mask=attn_mask)   # (batch x source_length x d_e)
        output = self.posifeedforward(output)                       # (batch x source_length x d_e)
        return output
    
    
class MultiHeadAttn(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttn, self).__init__()
        self.args = args
        self.proj_q = nn.Linear(args.d_e, args.d_q * args.num_heads)
        self.proj_k = nn.Linear(args.d_e, args.d_q * args.num_heads)
        self.proj_v = nn.Linear(args.d_e, args.d_q * args.num_heads)
        self.mixheads = nn.Linear(args.d_q * args.num_heads, args.d_e)
        self.layernorm = nn.LayerNorm(args.d_e)
    def forward(self, pre_q, pre_k, mask=None):
        # pre_q: (batch x q_len x d_e), pre_k: (batch x k_len x d_e)
        q_len, k_len = pre_q.shape[1], pre_k.shape[1]
        q = self.proj_q(pre_q).view(-1, q_len, self.args.num_heads, self.args.d_q).transpose(1, 2)   # (batch x num_heads x q_len x d_q)
        k = self.proj_k(pre_k).view(-1, k_len, self.args.num_heads, self.args.d_q).transpose(1, 2)   # (batch x num_heads x k_len x d_q)
        v = self.proj_v(pre_k).view(-1, k_len, self.args.num_heads, self.args.d_q).transpose(1, 2)   # (batch x num_heads x k_len x d_q)
        residual = pre_q
        
        # ScaledDotAttention
        attention = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.args.d_q)   # (batch x num_heads x q_len x k_len)
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(1).expand(-1, self.args.num_heads, -1, -1), -1e9)
            del mask
        attention = nn.Softmax(dim=-1)(attention)
        context = torch.matmul(attention, v)       # (batch x num_heads x q_len x d_q)
        context = context.transpose(1, 2).contiguous().view(-1, q_len, self.args.d_q * self.args.num_heads)
        context = self.mixheads(context)       # (batch x q_len x d_e)
        
        return self.layernorm(residual + context)
        
class PosiFeedForward(nn.Module):
    def __init__(self, args):
        super(PosiFeedForward, self).__init__()
        self.layers = nn.Sequential(nn.Conv1d(args.d_e, args.d_h, 1), nn.ReLU(), nn.Conv1d(args.d_h, args.d_e, 1))
        self.layernorm = nn.LayerNorm(args.d_e)
    def forward(self, input):
        # input: (batch x q_len x d_e)
        residual = input
        output = input.transpose(1, 2)   # (batch x d_e x q_len)
        output = self.layers(output).transpose(1, 2)   # (batch x q_len x d_e)
        output = self.layernorm(residual + output)
        return output
    
class PositionEmbedding():
    def __init__(self, args):
        def fill_poswise(dim_i):
            func = torch.sin if dim_i % 2 == 0 else torch.cos
            return func(torch.Tensor([pos/np.power(10000, 2 * (dim_i // 2) / args.d_e) for pos in range(1, 1 + args.max_length)]))
        table = torch.stack([fill_poswise(dim_i) for dim_i in range(1, 1 + args.d_e)]).unsqueeze(0)   # (1 x d_e x max_length)
        self.table = table.transpose(1, 2).to(args.device)   # (1 x max_length x d_e)
    def __call__(self, input):
        # input: (batch x length x d_e)
        len = input.shape[1]
        return input + self.table[:, :len]   # (batch x length x d_e)

        
    
    
    