import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # padding_idx表示用于填充的参数索引

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)
        # text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)  # [batch size, 1]


class WideCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # padding_idx表示用于填充的参数索引

        conv_list = []
        for i, fs in enumerate(filter_sizes):
            if i == 0:
                conv_list.append(
                    nn.Conv2d(in_channels=1, out_channels=n_filters,
                              kernel_size=(fs, embedding_dim), padding='same'  # 宽卷积，保持输入输出长度和维度一致
                              ))
            else:
                conv_list.append(
                    nn.Conv2d(in_channels=n_filters, out_channels=n_filters,
                              kernel_size=(fs, embedding_dim), padding='same'
                              ))

        self.convs = nn.ModuleList(conv_list)

        # self.filter_num = len(filter_sizes)
        self.fc = nn.Linear(embedding_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)
        # text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        for i, conv in enumerate(self.convs):
            if i == 0:
                h_i_pre = F.relu(conv(embedded))
                # print(h_i_pre.shape)  # [batch size, n_filters, sent len, emb dim]
            else:
                h_i = F.relu(conv(h_i_pre))
                h_i_pre = h_i

        # h_i = [batch size, n_filters, sent len, emb dim]

        h_avg = self.dropout(torch.mean(h_i, dim=2).squeeze(1))   # 当n_filters=1时才可squeeze
        # h_avg = [batch size, emb dim]

        return self.fc(h_avg)  # [batch size, 1]


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_dim, pad_idx):
        super().__init__()

        # self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_weight_matrix), freeze=True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_dim)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]
        # text = self.dropout(text)
        # text: seq, batch, embedding_dim
        output, (h_n, c_n) = self.rnn(embedded)

        # output = [batch size, sent len, hidden size]
        # h_n / c_n = [batch size, 1, hidden size]

        output = output[:, -1, :]  # 取最后一个time step
        # [batch size, hidden size]

        return self.fc(output)  # [batch size, 1]

