import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import random
import os
import copy
import warnings

from utils import pad_and_sort_batch, preprocess_for_batch, pad_or_truncate

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# class InterAttention(nn.Module):
#     def __init__(self, method, hidden_size, dropout_p=0.0):
#         super().__init__()
#         self.method = method
#         self.hidden_size = hidden_size
#         self.dropout = nn.Dropout(dropout_p)
#         if self.method == 'general':
#             self.linear = nn.Linear(self.hidden_size*2, self.hidden_size*2)
#         elif self.method == 'concat':
#             self.linear = nn.Linear(self.hidden_size*4, self.hidden_size*2)
#             self.v = nn.Linear(self.hidden_size*2, 1)
            
#     def forward(self, Q, K, V, attn_mask=None):
#         if self.method == 'dot':
#             attn_weights = torch.bmm(K, Q).squeeze(2)
#             # attn_weights shape: (batchsize, total_length)
#         elif self.method == 'general':
#             energy = F.tanh(self.linear(K))
#             attn_weights = torch.bmm(energy, Q).squeeze(2)
#         elif self.method == 'concat':
#             Q.transpose_(1, 2)
#             Q_expand = Q.expand(-1, K.size(1), -1)
#             cat = torch.cat((Q_expand, K), 2)
#             # cat shape: (batchsize, stepsize, 4*self.hidden_size)
#             energy = F.tanh(self.linear(cat))
#             Q.transpose_(1, 2)
#             attn_weights = self.v(energy).squeeze(2)
#         if attn_mask is not None:
#             attn_weights.masked_fill_(attn_mask, -float('inf'))
#         attn_weights = self.dropout(attn_weights)
#         soft_attn_weights = F.softmax(attn_weights, 1)
#         if attn_mask is not None:
#             soft_attn_weights = soft_attn_weights.masked_fill(attn_mask, 0) # fill all zero sentence (nan)
#         context = torch.matmul(attn, V)
#         return context, soft_attn_weights



class ScaleDotProductAttention(nn.Module):
    
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(K.size(2)) 
        # scores shape: (batch_size, n_heads, sen_len(len_q), sen_len(len_k))
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -float('inf'))
        attn = F.softmax(scores, dim=-1)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, 0)
        attn = self.dropout(attn)
        # attn shape: (batch_size, n_heads, len_q, len_k)
        context = torch.matmul(attn, V)
        # context shape: (batch_size, n_heads, sen_len, d_v)
        return context, attn


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, d_k, d_v, n_heads, multihead_dropout=0.0, self_att_dropout=0.0):
        super().__init__()
        self.d_model, self.d_k, self.d_v, self.n_heads = d_model, d_k, d_v, n_heads
        self.linear_Q = nn.Linear(d_model, d_k*n_heads)
        self.linear_K = nn.Linear(d_model, d_k*n_heads)
        self.linear_V = nn.Linear(d_model, d_v*n_heads)
        self.dropout = nn.Dropout(multihead_dropout)
        self.attn = ScaleDotProductAttention(self_att_dropout)
        self.out_linear = nn.Linear(d_v*n_heads, d_model)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        
    def forward(self, Q, K, V, attn_mask=None):
        residual = Q
        batch_size, sen_len, _ = Q.size()
        q_heads = self.linear_Q(Q).view(batch_size, sen_len, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.linear_K(K).view(batch_size, sen_len, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.linear_V(V).view(batch_size, sen_len, self.n_heads, self.d_v).transpose(1, 2)
        context, _ = self.attn(q_heads, k_heads, v_heads, attn_mask=attn_mask)
        # context shape: (batch_size, n_heads, sen_len, d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, sen_len, self.n_heads*self.d_v)
        # context shape: (batch_size, sen_len, d_model)
        output = self.out_linear(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        #output shape: (batch_size, sen_len, d_model)
        return output


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, d_k, d_v, n_heads, multihead_dropout=0.0, self_att_dropout=0.0):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, multihead_dropout, self_att_dropout)
        
    def forward(self, inputs, attn_mask=None):
        firstlayer_out = self.multihead_attention(inputs, inputs, inputs, attn_mask)
        return firstlayer_out


class TransformerEncoder(nn.Module):
    
    def __init__(self, n_layers, d_model, d_k, d_v, n_heads, multihead_dropout=0.0, transformer_dropout=0.0, self_att_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(transformer_dropout)
        self.encode_layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads, multihead_dropout, self_att_dropout) for _ in range(n_layers)])
        
        
    def padding_mask(self, Q, K):
        # inpus shape: (batch_size, sen_len)
        len_k, len_q = K.size(1), Q.size(1)
        pad_mask = K.eq(0)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, len_q, len_k)
        return pad_mask
        
    def forward(self, inputs, raw_inputs):
        pad_mask = self.padding_mask(raw_inputs, raw_inputs)
        for layer in self.encode_layers:
            inputs = layer(inputs, attn_mask=pad_mask)
        inputs = self.dropout(inputs)
        return inputs



class EncoderBiLSTM(nn.Module):
    def __init__(self, hidden_size=16, output_size=1, rnn_dropout=0.3, embedding_dropout=0.3, embedding_dim=100, vocab_size=10000, embedding=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding).to(DEVICE), freeze=False)
            self.vocab_size, self.embedding_dim = embedding.shape
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        #self.layer_norm = nn.LayerNorm(2*self.hidden_size, elementwise_affine=True)
             
    def forward(self, X, lengths):
        total_length = X.size(1)  # get the max sequence length
        embedded = self.embedding(X)
        embedded = self.embedding_dropout(embedded)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        packed_output, (h_n, c_n) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        output = self.rnn_dropout(output)
        # output shape: (batchsize, total_length, 2*hidden_size)
        # h_n shape: (num_direction, batchsize, hidden_size)
        
        c_n = c_n.transpose(0, 1).contiguous().view(-1, 1, 2*self.hidden_size)
        h_n = h_n.transpose(0, 1).contiguous().view(-1, 1, 2*self.hidden_size)
        #h_n = self.layer_norm(h_n)
        h_n.transpose_(1, 2)
        c_n.transpose_(1, 2)
        #output = self.layer_norm(output)
        #return output, c_n
        return output, h_n


class InterAttention(nn.Module):
    def __init__(self, method, hidden_size, dropout_p=0.0):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_p)
        if self.method == 'no_query':
            self.linear = nn.Linear(self.hidden_size*2, 1)

        if self.method == 'general':
            self.linear = nn.Linear(self.hidden_size*2, self.hidden_size*2)
            
        elif self.method == 'concat':
            self.linear = nn.Linear(self.hidden_size*4, self.hidden_size*2)
            self.v = nn.Linear(self.hidden_size*2, 1)
            
            
    def forward(self, encoder_output, hidden, attn_mask=None):
        if self.method == 'no_query':
            attn_weights = self.linear(encoder_output).squeeze(2)

        elif self.method == 'dot':
            attn_weights = torch.bmm(encoder_output, hidden).squeeze(2)
            # attn_weights shape: (batchsize, total_length)

        elif self.method == 'general':
            energy = F.tanh(self.linear(encoder_output))
            attn_weights = torch.bmm(energy, hidden).squeeze(2)
        
        elif self.method == 'concat':
            hidden.transpose_(1, 2)
            hidden_expand = hidden.expand(-1, encoder_output.size(1), -1)
            cat = torch.cat((hidden_expand, encoder_output), 2)
            # cat shape: (batchsize, stepsize, 4*self.hidden_size)
            energy = F.tanh(self.linear(cat))
            hidden.transpose_(1, 2)
            attn_weights = self.v(energy).squeeze(2)
        
        if attn_mask is not None:
            attn_weights.masked_fill_(attn_mask, -float('inf'))
        attn_weights = self.dropout(attn_weights)
        soft_attn_weights = F.softmax(attn_weights, 1)
        return soft_attn_weights


class SIAttentionBiLSTM(nn.Module):
    
    def __init__(self, n_layers, d_model, d_k, d_v, n_heads, hidden_size=16, output_size=1, multihead_dropout=0.0, rnn_dropout=0.0, embedding_dropout=0.0, context_dropout=0.0, transformer_dropout=0.0, inter_att_dropout=0.0, self_att_dropout=0.0, embedding_dim=100, vocab_size=10000, embedding=None, method='dot'):
        super().__init__()
        self.lstm_encoder = EncoderBiLSTM(hidden_size, output_size, rnn_dropout, embedding_dropout, embedding_dim, vocab_size, embedding)
        self.transformer_encoder = TransformerEncoder(n_layers, d_model, d_k, d_v, n_heads,multihead_dropout=multihead_dropout, transformer_dropout=transformer_dropout, self_att_dropout=self_att_dropout)
        self.hidden = torch.empty(1, hidden_size*2, 1).uniform_(-1, 1).requires_grad_(requires_grad=True).to(DEVICE)
        self.attn = InterAttention(method, hidden_size, dropout_p=inter_att_dropout)
        self.context_dropout = nn.Dropout(context_dropout)
        self.out = nn.Linear(2*hidden_size, output_size)
        
    def forward(self, X, lengths):
        sen_len = X.size(1)
        lstm_output, hidden_state = self.lstm_encoder(X, lengths)
        encoder_output = self.transformer_encoder(lstm_output, X)
        attn_weights = self.attn(encoder_output, self.hidden.expand(X.size(0),-1,-1), attn_mask=X.eq(0))
        #attn_weights = self.attn(lstm_output, hidden_state, attn_mask=X.eq(0))
        context = torch.bmm(encoder_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        context = self.context_dropout(context)
        # context shape: (batchsize, 2*hidden_size)
        final_output = self.out(context)
        return final_output