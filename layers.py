import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedder import WordEmbedder, PositionEncoder


def scaled_dot_product(q, k, v, mask=None):
    # get the dimension of the embedded vector (1-d)
    d_k = q.size()[-1]
    # compute Q(K^t)
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1) # dim = -1 means the last dimension
    # compute {soft(QK^t/sqrt(d))}V
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads"
        # embed_dim is the stacked embedded vectors
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # head_dim is the dimension of each individual embedding head (like channel in cnn)
        # num_heads ~ channels in cnn
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(input_dim, embed_dim)  # q,k,v, each has dim = embed_dim
        self.v_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # original transformer initailization, see PyTorch documentation
        # bias is initialized as zeros
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    # this forward operation is different from
    # the reference: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#The-Transformer-architecture
    # this implementation makes multihead attention in decoder much easier
    # because the input (forward) of mutlihead attention in decoder is different
    # unlike in the encoder the input for q,k,v inputs are the same
    def forward(self, xq, xv, xk, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = xq.size()
        # separate q, k, v from linear output
        # q, k, v are learned separately
        q = self.q_proj(xq).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(xv).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(xk).view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)

        # determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2, dropout=0.8):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        *** feedforward architecture can be customized
        """
        super().__init__()

        # attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # attention part
        attn_out = self.self_attn(x, x, x, mask=None)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout2(linear_out)
        x = self.norm2(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward=2, dropout=0.8):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        Feedforward architecture can be customized
        """

        super().__init__()

        # attention layer
        self.self_attn1 = MultiheadAttention(input_dim, input_dim, num_heads)
        self.self_attn2 = MultiheadAttention(input_dim, input_dim, num_heads)

        # two-layer MLP
        self.linear_net = nn.Sequential(nn.Linear(input_dim, dim_feedforward),
                                        nn.Dropout(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(dim_feedforward, input_dim))

        # layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask=None, tar_mask=None):
        # attention part
        x = self.norm1(x)
        x = x + self.dropout1(self.self_attn1(x, x, x, mask=tar_mask))
        x = self.norm2(x)
        x = x + self.dropout2(self.self_attn2(x, e_outputs, e_outputs, src_mask))
        x = self.norm3(x)
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout3(linear_out)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, vocab_size, embed_dim, num_heads, dim_feedforward=2, dropout=0.8):
        super().__init__()
        self.embed = WordEmbedder(vocab_size, embed_dim)
        self.pE = PositionEncoder(embed_dim)
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, num_heads, dim_feedforward, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pE(x)
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, vocab_size, embed_dim, num_heads, dim_feedforward=2, dropout=0.8):
        super().__init__()
        self.embed = WordEmbedder(vocab_size, embed_dim)
        self.pE = PositionEncoder(embed_dim)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads, dim_feedforward, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, e_outputs, src_mask=None, tar_mask=None):
        x = self.embed(x)
        x = self.pE(x)
        for l in self.layers:
            x = l(x, e_outputs, src_mask, tar_mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
