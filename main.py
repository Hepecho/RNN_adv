from model import WideCNN, RNN, CNN
from trainer import train_model
import torch.optim as optim
import torch.nn as nn
import argparse
import torch
from os.path import join as ospj
import os
import random
import numpy as np
import dill
from runx.logx import logx

from dataloader import prepare_data, text_transform, label_transform
import dataloader
from utils import freeze_layer
# from torchstat import stat


def predict_sentiment(model, sentence, device, min_len=5):
    model.to(device)
    model.eval()
    with torch.no_grad():
        text = text_transform(sentence)
        # print(len(text))
        if len(text) - 2 < min_len:
            text = text[1:-1] + [VOCAB['<PAD>']] * (min_len - len(text) + 2)
        text = torch.tensor(text)
        # print(len(text))
        text = text.unsqueeze(1)
        # [sent len, 1]
        text = text.to(device)
        output = model(text).squeeze(1)
        prediction = torch.sigmoid(output)
        return prediction.item(), torch.round(prediction).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab 3.1 RNN_adv')
    parser.add_argument('--mode', type=str, default='train',
                        help='train, eval')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max_vocab_size', type=int, default=25000,
                        help='max_vocab_size (default: 25000)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--filters', type=int, default=100,
                        help='number of filters to train (default: 100)')
    parser.add_argument('--filter_sizes', nargs='+', default=[3, 4, 5],
                        help='h of filters (default: [3, 4, 5])')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='word vector dim (default: 300)')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='lstm hidden dim (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001 for adam; 0.1 for SGD)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='disables CUDA training')
    parser.add_argument('--embedding', default='random', type=str,
                        help="embedding type ['random', 'glove']")
    parser.add_argument('--freeze_embedding', action='store_true',
                        help='freeze embedding layer, if false then fine-tuning embedding layer')
    parser.add_argument('--preprocess_data', action='store_true',
                        help='preprocess data, if false then load preprocessed data')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--logdir', type=str, default='log',
                        help='target log directory')
    parser.add_argument('--ckptdir', type=str, default='model',
                        help='checkpoint directory')
    parser.add_argument('--model', type=str, default='cnn',
                        help='cnn, cnn_wide, rnn')

    args = parser.parse_args()

    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    print(device)

    os.makedirs(ospj(args.ckptdir, args.model), exist_ok=True)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    if args.mode == 'train':
        logx.initialize(logdir=ospj(args.logdir, args.model), coolname=False, tensorboard=False)
        VOCAB, glove, train_dataloader, valid_dataloader, test_dataloader = prepare_data(args)
        dataloader.VOCAB = VOCAB
        # if args.embedding == 'glove':
        #     dataloader.glove = glove

        # 打印字典长度，确定是同一个预处理数据集
        print(len(VOCAB))

        if args.model == 'cnn':
            model = CNN(len(VOCAB), args.embedding_dim, args.filters, args.filter_sizes,
                        output_dim=1, dropout=args.dropout, pad_idx=VOCAB['<PAD>'], embedding=args.embedding,
                        VOCAB=VOCAB, glove=glove)
        elif args.model == 'rnn':
            model = RNN(len(VOCAB), args.embedding_dim, args.hidden_size,
                        output_dim=1, pad_idx=3)
        else:
            model = WideCNN(len(VOCAB), args.embedding_dim, args.filters, args.filter_sizes,
                        output_dim=1, dropout=args.dropout, pad_idx=3)

        if args.freeze_embedding:
            freeze_layer(model.embedding)

        print("model.embedding.weight", model.embedding.weight)
        # 仅设置fc的w参数正则化
        fc_w = (param for name, param in model.fc.named_parameters() if name[-4:] != 'bias')
        fc_w_map = list(map(id, model.fc.parameters()))  # id函数返回对象的“标识值”，整数
        others_list = (param for name, param in model.named_parameters() if id(param) not in fc_w_map)
        # print(others_list)
        # exit()
        parameters = [{'params': fc_w, 'weight_decay': 2},
                      {'params': others_list}]
        optimizer = optim.Adadelta(parameters, lr=args.lr, weight_decay=0)

        criterion = nn.BCEWithLogitsLoss()  # 将 Sigmoid 层和 BCELoss 合并

        model = train_model(args, model, optimizer, criterion,
                            train_dataloader, valid_dataloader, test_dataloader, device)

        print("model.embedding.weight", model.embedding.weight)

    else:
        best_model_path = ospj(args.ckptdir, args.model, 'best_model.pt')
        with open('./data/vocab.pkl', 'rb') as f:
            VOCAB = dill.load(f)
        dataloader.VOCAB = VOCAB
        if args.model == 'cnn':
            model = CNN(len(VOCAB), args.embedding_dim, args.filters, args.filter_sizes,
                             output_dim=1, dropout=args.dropout, pad_idx=3)
        elif args.model == 'rnn':
            model = RNN(len(VOCAB), args.embedding_dim, args.hidden_size,
                        output_dim=1, pad_idx=3)
        else:
            model = WideCNN(len(VOCAB), args.embedding_dim, args.filters, args.filter_sizes,
                            output_dim=1, dropout=args.dropout, pad_idx=3)
        model.load_state_dict(torch.load(best_model_path))
        with open('remark.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        print(predict_sentiment(model, content, device))

        # model.eval()
        # stat(model, input_size=(1, 200, 16))

        # "nice film, greate film, terrific film"
        # "bad film, lousy film, stupid film"
