import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import warnings
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
             
    def forward(self, X):
        embedded = self.embedding(X)
        embedded = self.embedding_dropout(embedded)
        output, (h_n, c_n) = self.lstm(embedded)
        output = self.rnn_dropout(output)
        # output shape: (batchsize, total_length, 2*hidden_size)
        # h_n shape: (num_direction, batchsize, hidden_size)
        h_n = h_n.transpose(0, 1).contiguous().view(-1, 1, 2*self.hidden_size)
        h_n.transpose_(1, 2)
        return output, h_n


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
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
        soft_attn_weights = F.softmax(attn_weights, 1)
        if attn_mask is not None:
            soft_attn_weights = soft_attn_weights.masked_fill(attn_mask, 0) # fill all zero sentence (nan)
        return soft_attn_weights
    
class HAN(nn.Module):
    def __init__(self, hidden_size=16, output_size=1, rnn_dropout=0.1, embedding_dropout=0.3, word_dropout=0.1, sent_dropout=0.1, embedding_dim=100, vocab_size=10000, embedding=None, method='dot'):
        super().__init__()
        self.word_encoder = EncoderBiLSTM(hidden_size, output_size, rnn_dropout, embedding_dropout, embedding_dim, vocab_size, embedding)
        self.word_hidden = torch.empty(1, hidden_size*2, 1).uniform_(-1, 1).requires_grad_(requires_grad=True).to(DEVICE)
        #self.word_hidden = nn.Parameter(torch.empty(1, hidden_size*2, 1).uniform_(-1, 1).requires_grad_(requires_grad=True).to(DEVICE))
        self.word_attn = Attention(method, hidden_size)
        self.word_dropout = nn.Dropout(word_dropout)
        self.sent_lstm = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        #self.sent_hidden = nn.Parameter(torch.empty(1, hidden_size*2, 1).uniform_(-1, 1).requires_grad_(requires_grad=True).to(DEVICE))
        self.sent_hidden = torch.empty(1, hidden_size*2, 1).uniform_(-1, 1).requires_grad_(requires_grad=True).to(DEVICE)
        self.sent_attn = Attention(method, hidden_size)
        self.sent_dropout = nn.Dropout(sent_dropout)
        self.out = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, X):
        batch_size, sen_nums, sen_len = X.size()
        X = X.view(batch_size*sen_nums, sen_len)
        encoder_output, hidden_state = self.word_encoder(X)
        # encoder_output shape:(batchsize*sen_nums, sen_len, num_directions * hidden_size)
        word_attn_weights = self.word_attn(encoder_output, self.word_hidden.expand(X.size(0),-1,-1), attn_mask=X.eq(0))
        word_context = torch.bmm(encoder_output.transpose(1, 2), word_attn_weights.unsqueeze(2)).squeeze(2)
        context = self.word_dropout(word_context)
        # context shape: (batchsize*sen_nums, 2*hidden_size)
        context = context.view(batch_size, sen_nums, -1)
        context, _ = self.sent_lstm(context)
        sent_attn_weights = self.sent_attn(context, self.sent_hidden.expand(context.size(0),-1,-1), attn_mask=X.view(batch_size, sen_nums, sen_len).sum(dim=2, keepdim=False).eq(0))
        context = torch.bmm(context.transpose(1, 2), sent_attn_weights.unsqueeze(2)).squeeze(2)
        context = self.sent_dropout(context)
        final_output = self.out(context)
        return final_output