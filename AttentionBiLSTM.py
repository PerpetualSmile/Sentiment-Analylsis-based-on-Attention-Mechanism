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
        return output, c_n, h_n


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
        return soft_attn_weights


class AttnBiLSTM(nn.Module):
    
    def __init__(self, hidden_size=16, query='h', output_size=1, rnn_dropout=0.3, embedding_dropout=0.3, context_dropout=0.3, embedding_dim=100, vocab_size=10000, embedding=None, method='dot'):
        super().__init__()
        self.query = query
        self.encoder = EncoderBiLSTM(hidden_size, output_size, rnn_dropout, embedding_dropout, embedding_dim, vocab_size, embedding)
        self.hidden = torch.empty(1, hidden_size*2, 1).uniform_(-1, 1).requires_grad_(requires_grad=True).to(DEVICE)
        self.attn = Attention(method, hidden_size)
        self.context_dropout = nn.Dropout(context_dropout)
        self.out = nn.Linear(2*hidden_size, output_size)
        
    def forward(self, X, lengths):
        encoder_output, c_n, h_n = self.encoder(X, lengths)
        if self.query is 'p':
            attn_weights = self.attn(encoder_output, self.hidden.expand(X.size(0),-1,-1), attn_mask=X.eq(0))
        elif self.query is 'h':
            attn_weights = self.attn(encoder_output, h_n, attn_mask=X.eq(0))
        elif self.query is 'c':
            attn_weights = self.attn(encoder_output, c_n, attn_mask=X.eq(0))
        context = torch.bmm(encoder_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        context = self.context_dropout(context)
        # context shape: (batchsize, 2*hidden_size)
        final_output = self.out(context)
        return final_output


