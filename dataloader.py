import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import GloVe
from torchtext.vocab import vocab

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split

import dill
import time
from runx.logx import logx


def yield_tokens(train_iter):
    for (label, line) in train_iter:
        yield line.strip().split()


# 分词器
tokenizer = get_tokenizer('basic_english')

VOCAB = None
# random
text_transform = lambda x: [VOCAB['<BOS>']] + [VOCAB[token] for token in tokenizer(x)] + [VOCAB['<EOS>']]
label_transform = lambda x: 1 if x == 'pos' else 0

glove = None

def collate_batch_random(batch):
    """
    padding_value将这个批次的句子全部填充成一样的长度，padding_value=word_vocab['<PAD>']=3
    """
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

def collate_batch_glove(batch):
    """
    padding_value将这个批次的句子全部填充成一样的长度，padding_value=word_vocab['<PAD>']=3
    """
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=VOCAB['<PAD>'])


def prepare_data(args):
    BATCH_SIZE = args.batch_size

    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Start Prepare Data [{}]===================='.format(localtime))

    if args.preprocess_data:
        # 获取数据
        train_data, test_data = IMDB(root='data/aclImdb', split=('train', 'test'))

        # 切分训练集
        train_dataset = to_map_style_dataset(train_data)
        # print(next(iter(test_data)))
        num_train = int(len(train_dataset) * 0.90)
        # print(len(train_dataset)) 25000
        # exit()
        train_data, valid_data = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
        # print(len(valid_data)) 2500

        # 创建词表
        if args.embedding != 'random':
            glove = GloVe(name='6B', dim=300)
            # 返回的实例为Vectors类，主要有以下三个属性：
            # stoi: 词到索引的字典：
            # itos: 一个列表，索引到词的映射；
            # vectors: 词向量
            # word_vectors = glove.vectors
            VOCAB_dict = glove.stoi  # 该VOCAB_dict是dict类，而random得到的VOCAB是Vocab类对象

            VOCAB = vocab(VOCAB_dict, specials=['<unk>', '<BOS>', '<EOS>', '<PAD>'])  # 转换成Vocab类
            VOCAB.set_default_index(VOCAB['<unk>'])

            collate_batch = collate_batch_glove

        else:
            VOCAB = build_vocab_from_iterator(yield_tokens(train_data), min_freq=10,
                                              specials=['<unk>', '<BOS>', '<EOS>', '<PAD>'])  # 建立词表
            VOCAB.set_default_index(VOCAB['<unk>'])
            collate_batch = collate_batch_random

        with open('./data/vocab.pkl', 'wb') as f:
            dill.dump(VOCAB, f)

        train_dataloader = DataLoader(list(train_data), batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_batch)
        valid_dataloader = DataLoader(list(valid_data), batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_batch)
        test_dataloader = DataLoader(list(test_data), batch_size=BATCH_SIZE,
                                     shuffle=True, collate_fn=collate_batch)

        with open('./data/train_dataloader_save.pkl', 'wb') as f:
            dill.dump(train_dataloader, f)
        with open('./data/valid_dataloader_save.pkl', 'wb') as f:
            dill.dump(valid_dataloader, f)
        with open('./data/test_dataloader_save.pkl', 'wb') as f:
            dill.dump(test_dataloader, f)
    else:
        if args.embedding != 'random':
            glove = GloVe(name='6B', dim=300)
        with open('./data/vocab.pkl', 'rb') as f:
            VOCAB = dill.load(f)
        with open('./data/train_dataloader_save.pkl', 'rb') as f:
            train_dataloader = dill.load(f)
        with open('./data/valid_dataloader_save.pkl', 'rb') as f:
            valid_dataloader = dill.load(f)
        with open('./data/test_dataloader_save.pkl', 'rb') as f:
            test_dataloader = dill.load(f)

    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Finish Prepare Data [{}]===================='.format(localtime))

    return VOCAB, glove, train_dataloader, valid_dataloader, test_dataloader

