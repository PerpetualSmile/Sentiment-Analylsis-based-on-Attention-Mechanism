import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import warnings
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AvgBiLSTM(nn.Module):
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
        self.linear = nn.Linear(2*self.hidden_size, output_size)
             
    def forward(self, X, lengths):
        total_length = X.size(1)  # get the max sequence length
        embedded = self.embedding(X)
        embedded = self.embedding_dropout(embedded)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        packed_output, (h_n, c_n) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        output = self.rnn_dropout(output)
        output = torch.mean(output, dim=1, keepdim=False)
        output = self.linear(output)
        return output
