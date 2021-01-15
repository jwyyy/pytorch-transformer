import torch
from transformer import Transformer
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
