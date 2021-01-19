import torch
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
        np_mask = np_mask.cuda()
    return np_mask


def create_masks(batch, EN_TEXT, FR_TEXT):
    src_seq = batch.English.transpose(0, 1)
    src_pad = EN_TEXT.vocab.stoi['<pad>']  # creates mask with 0s wherever there is padding in the input
    src_mask = (src_seq != src_pad).unsqueeze(1)

    tar_seq = batch.French.transpose(0, 1)
    tar_pad = FR_TEXT.vocab.stoi['<pad>']
    tar_mask = (tar_seq != tar_pad).unsqueeze(1)
    size = tar_seq.size(1)  # get seq_len for matrix
    nopeak_mask = np.triu(np.ones(1, size, size), k = 1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    tar_mask = tar_mask & nopeak_mask

    return src_mask, tar_mask
