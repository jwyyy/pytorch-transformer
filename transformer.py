# reference post: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# reference : https://github.com/SamLynnEvans/Transformer
# reference : https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html


# Note:
# inputs for MultiHead Attention in encoders and decoders are different.
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math 


def scaled_dot_product(q, k, v, mask = None):
	# get the dimension of the embedded vector (1-d)
	d_k = q.size()[-1]
	# compute Q(K^t)
	attn_logits = torch.matmul(q, k.transpose(-2,-1)) # transpose the last two dimensions
	attn_logits = attn_logits / math.sqrt(d_k)
	if mask is not None:
		attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
	attention = F.softmax(attn_lgots, dim=-1)
	# compute {soft(QK^t/sqrt(d))}V
	values = torch.matmul(attention, v)
	return values, attention

class MultiheadAttention(nn.Module):
	def __init__(self, input_dim, embed_dim, num_heads):
		pass
	
	def _reset_parameters(self):
		pass

	def forward(self, x, mask = None, return_attention = False):
		pass

