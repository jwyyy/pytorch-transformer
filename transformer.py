# reference post: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# reference : https://github.com/SamLynnEvans/Transformer
# reference : https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html


# Note:
# inputs for MultiHead Attention in encoders and decoders are different.
# 

import torch
import torch.nn as nn
from layers import TransformerEncoder, TransformerDecoder


class Transformer(nn.Module):
	def __init__(self, src_vocab, tar_vocab, embed_dim, num_layers, num_heads):
		super().__init__()
		self.encoder = TransformerEncoder(num_layers, src_vocab, embed_dim, num_heads)
		self.decoder = TransformerDecoder(num_layers, tar_vocab, embed_dim, num_heads)
		self.out = nn.Linear(embed_dim, tar_vocab)

	def forward(self, src, tar, src_mask, tar_mask):
		e_outputs = self.encoder(src, src_mask)
		d_outputs = self.decoder(tar, e_outputs, src_mask, tar_mask)
		output = self.out(d_outputs)

		return output


