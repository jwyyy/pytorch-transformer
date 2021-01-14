import torch
import math
import torch.nn as nn
from torch.autograd import Variable


class WordEmbedder(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim)

    def forward(self, word):
        return self.embedder(word)


class PositionEncoder(nn.Module):

    def __init__(self, embed_dim, max_seq_len=80):
        # max_seq_len is the length of the sentence
        super().__init__()
        self.embed_dim = embed_dim
        posEncoder = torch.zeros(max_seq_len, embed_dim)
        for i in range(max_seq_len):
            for d in range(0, embed_dim, 2):
                posEncoder[i, d] = math.sin(i / (10000 ** ((2 * d) / embed_dim)))
                posEncoder[i, d + 1] = math.cos(i / (10000 ** ((2 * (d + 1)) / embed_dim)))
        posEncoder.unsequeeze(0)
        self.register_buffer('posEncoder', posEncoder)

    def forward(self, x):
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)  # [Batch, Seq_Len, #, #
        pE = Variable(self.posEncoder[:, :seq_len], require_grad=False)
        if x.is_cuda():
            pE.cuda()
        x = x + pE
        return x
