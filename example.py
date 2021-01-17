
# reference: https://towardsdatascience.com/
# how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95

import pandas as pd
import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
from functools import partial
from sklearn.model_selection import train_test_split


en = spacy.load('en')
fr = spacy.load('fr')

europarl_en = open('fr-en/europarl-v7.fr-en.en', encoding='utf-8').read().split('\n')
europarl_fr = open('fr-en/europarl-v7.fr-en.fr', encoding='utf-8').read().split('\n')


def tokenizer(tokzer, sentence):
    return [tok.text for tok in tokzer.tokenizer(sentence)]


EN_TEXT = Field(tokenize=partial(tokenizer, en))
FR_TEXT = Field(tokenize=partial(tokenizer, fr), init_token='<sos>', eos_token='<eos>')

# transform data into csv first

raw_data = {'Englsih': [line for line in europarl_en],
            'French' : [line for line in europarl_fr]}
df = pd.DataFrame(raw_data, columns=['English', 'French'])
df['en_len'] = df['English'].str.count(' ')
df['fr_len'] = df['French'].str.count(' ')
df = df.query('en_len < 80 && fr_len < 80')
df = df.query('en_len < fr_len * 1.5 & fr_len < en_len * 1.5')

train, val = train_test_split(df, test_size=0.1)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)

data_fields = [('English', EN_TEXT), ('French', FR_TEXT)]
train, val = TabularDataset.splits(path='./', train='train.csv', validation='val.csv',
                                   format='csv', fields=data_fields)

EN_TEXT.build_vocab(train, val)
FR_TEXT.build_vocab(train, val)

train_itr = BucketIterator(train, batch_size=20, sort_key=lambda x: len(x.French), shuffle=True)

# will implement a torchtext version tomorrow
