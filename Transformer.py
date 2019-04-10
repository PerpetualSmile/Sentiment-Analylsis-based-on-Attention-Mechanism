import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import warnings
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


d_model = 512 # embedding size
n_heads = 8 # number of heads in Multihead Attention
d_k = d_v = d_model//n_heads
d_ff = 2048 # Postion-wise Feed-Forward-Forward Networks inner layer dims
n_layers = 6 # number of encoder layer

class PE_add_Embedding(nn.Module):
    
    def __init__(self, embedding_dim=512, vocab_size=10000, embedding=None, maxpos=5000, dropout_p=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        
        if embedding is None:
            self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)
            self.vocab_size, self.embedding_dim = embedding.shape
            
        self.pos_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.getPE(maxpos+1, self.embedding_dim)), freeze=True)
        self.dropout = nn.Dropout(self.dropout_p)
        
        
    def getPE(self, n_position, d_model):
        pos_encode = np.array([
            [pos / np.power(10000, 2*(i//2)/d_model) for i in range(d_model)]
            for pos in range(n_position)
        ])
        pos_encode[:, 0::2] = np.sin(pos_encode[:, 0::2])
        pos_encode[:, 1::2] = np.cos(pos_encode[:, 1::2])
        return pos_encode
    
    def forward(self, inputs):
        batch_size, sen_len = inputs.size()
        outputs = self.word_embedding(inputs) + self.pos_embedding(torch.range(1, sen_len, dtype=torch.int64).view(1, -1).expand(batch_size, -1).to(DEVICE))
        outputs = self.dropout(outputs)
        return outputs


class ScaleDotProductAttention(nn.Module):
    
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(d_k) 
        # scores shape: (batch_size, n_heads, sen_len(len_q), sen_len(len_k))
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -float('inf'))
        attn = F.softmax(scores, dim=-1)
        # attn shape: (batch_size, n_heads, len_q, len_k)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        # context shape: (batch_size, n_heads, sen_len, d_v)
        return context, attn


class MultiHeadAttention(nn.Module):
    
    def __init__(self, multihead_dropout=0.0, att_dropout=0.0):
        super().__init__()
        self.linear_Q = nn.Linear(d_model, d_k*n_heads)
        self.linear_K = nn.Linear(d_model, d_k*n_heads)
        self.linear_V = nn.Linear(d_model, d_v*n_heads)
        self.dropout = nn.Dropout(multihead_dropout)
        self.attn = ScaleDotProductAttention(dropout_p=att_dropout)
        self.out_linear = nn.Linear(d_v*n_heads, d_model)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        
    def forward(self, Q, K, V, attn_mask=None):
        residual = Q
        batch_size, sen_len, _ = Q.size()
        q_heads = self.linear_Q(Q).view(batch_size, sen_len, n_heads, d_k).transpose(1, 2)
        k_heads = self.linear_K(K).view(batch_size, sen_len, n_heads, d_k).transpose(1, 2)
        v_heads = self.linear_V(V).view(batch_size, sen_len, n_heads, d_v).transpose(1, 2)
        context, _ = self.attn(q_heads, k_heads, v_heads, attn_mask=attn_mask)
        # context shape: (batch_size, n_heads, sen_len, d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, sen_len, n_heads*d_v)
        # context shape: (batch_size, sen_len, d_model)
        output = self.out_linear(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        # output shape: (batch_size, sen_len, d_model)
        return output


class PoswiseFeedForward(nn.Module):
    
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        
    def forward(self, inputs):
        # inputs shape: (batch_size, sen_len, d_model)
        residual = inputs
        outputs = F.relu(self.conv1(inputs.transpose(1, 2)))
        outputs = self.conv2(outputs).transpose(1, 2)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + residual)
        # outputs shape == inputs shape
        return outputs


class EncoderLayer(nn.Module):
    
    def __init__(self, multihead_dropout=0.0, att_dropout=0.0, feedforward_dropout=0.0):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(multihead_dropout=multihead_dropout, att_dropout=att_dropout)
        self.postion_feedforward = PoswiseFeedForward(dropout_p=feedforward_dropout)
        
    def forward(self, inputs, attn_mask=None):
        firstlayer_out = self.multihead_attention(inputs, inputs, inputs, attn_mask)
        secondlayer_out = self.postion_feedforward(firstlayer_out)
        return secondlayer_out


class InterAttention(nn.Module):
    def __init__(self, method='dot'):
        super().__init__()
        self.method = method
        if self.method == 'general':
            self.linear = nn.Linear(d_model, d_model)
            
        elif self.method == 'concat':
            self.linear = nn.Linear(2*d_model, d_model)
            self.v = nn.Linear(d_model, 1)
            
            
    def forward(self, Q, K, V, attn_mask=None):
        if self.method == 'dot':
            attn_weights = torch.bmm(K, Q).squeeze(2)
            # attn_weights shape: (batchsize, total_length)
        
        elif self.method == 'general':
            energy = F.tanh(self.linear(K))
            attn_weights = torch.bmm(energy, Q).squeeze(2)
        
        elif self.method == 'concat':
            Q.transpose_(1, 2)
            Q_expand = Q.expand(-1, K.size(1), -1)
            cat = torch.cat((Q_expand, K), 2)
            # cat shape: (batchsize, sen_len, 2*d_model)
            energy = F.tanh(self.linear(cat))
            Q.transpose_(1, 2)
            attn_weights = self.v(energy).squeeze(2)

        if attn_mask is not None:
            attn_weights.masked_fill_(attn_mask, -float('inf'))
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(V.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights



class Transformer(nn.Module):
    
    def __init__(self, output_size=1, architecture = (6, 512, 2048, 8, 64, 64), method='dot', embedding_dim=d_model, vocab_size=10000, embedding=None, maxpos=5000, embedding_dropout=0.0, multihead_dropout=0.0, att_dropout=0.0, feedforward_dropout=0.0, final_dropout=0.0):
        super().__init__()
        global d_model, n_heads, d_k, d_v, d_ff, n_layers
        n_layers, d_model, d_ff, n_heads, d_k, d_v = architecture
        self.pos_add_word_embedding = PE_add_Embedding(
            embedding_dim=d_model, 
            vocab_size=vocab_size, 
            embedding=embedding,
            maxpos=maxpos,
            dropout_p=embedding_dropout
        )
        self.encode_layers = nn.ModuleList([EncoderLayer(multihead_dropout, att_dropout, feedforward_dropout) for _ in range(n_layers)])
        self.hidden = torch.empty(1, d_model, 1).uniform_(-1, 1).requires_grad_(requires_grad=True).to(DEVICE)
        self.inter_attn = InterAttention(method=method)
        self.dropout = nn.Dropout(final_dropout)
        self.fc = nn.Linear(d_model, output_size)
        
        
    def padding_mask(self, Q, K):
        # inpus shape: (batch_size, sen_len)
        len_k, len_q = K.size(1), Q.size(1)
        pad_mask = K.eq(0)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).expand(-1, n_heads, len_q, len_k)
        return pad_mask
        
    def forward(self, inputs):
        outputs = self.pos_add_word_embedding(inputs)
        pad_mask = self.padding_mask(inputs, inputs)
        for layer in self.encode_layers:
            outputs = layer(outputs, attn_mask=pad_mask)
        # outputs shape: (batch_size, sen_len, d_model)
        context, attn_weights = self.inter_attn(self.hidden.expand(inputs.size(0),-1,-1), outputs, outputs, attn_mask=inputs.eq(0))
        # context = torch.mean(outputs, dim=1, keepdim=False)
        # outputs shape: (batch_size, d_model)
        outputs = self.dropout(context)
        outputs = self.fc(outputs)
        #outputs shape: (batch_size, 1)
        return outputs
