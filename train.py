import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from data import getData
from mask import create_masks
from transformer import Transformer


def train_model(model, optimizer, train_itr, EN_TEXT, FR_TEXT, epochs = 10000, print_every = 100):

    # tell the model, we are training the model
    # b/c dropout, batch norm behave differently on the train and test procedures
    model.train()
    start = time.time()
    prev = start
    total_loss = 0

    for ep in range(epochs):
        for i, batch in enumerate(train_itr):
            src = batch.English.transpose(0,1)
            tar = batch.French.transpose(0,1)
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next

            tar_input = tar[:, :-1]
            targets = tar[:, 1:].contiguous().view(-1)

            src_mask, tar_mask = create_masks(batch, EN_TEXT, FR_TEXT)
            preds = model(src, tar_input, src_mask, tar_mask)

            optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets, ignore_index=t)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]

            if (i+1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,% ds per % d iters" % ((time.time() - start) // 60,
                                                                                           ep + 1,
                                                                                           i + 1,
                                                                                           loss_avg,
                                                                                           time.time() - prev,
                                                                                           print_every))
                total_loss = 0
                prev = time.time()


if __name__ == '__main__':
    d_model = 512
    heads = 8
    N = 6
    EN_TEXT, FR_TEXT, train_itr = getData()
    src_vocab = len(EN_TEXT.vocab)
    trg_vocab = len(FR_TEXT.vocab)

    model = Transformer(src_vocab, trg_vocab, d_model, N, heads)

    # model parameters are initialized in each block/layer (using _reset_parameters())

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    train_model(model, optimizer, train_itr, EN_TEXT, FR_TEXT)

