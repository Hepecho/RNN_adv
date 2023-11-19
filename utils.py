import torch
import time
import pandas as pd

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def save_data(cache, csv_path):
    colums = list(cache.keys())
    values = list(cache.values())
    values_T = list(map(list, zip(*values)))
    save = pd.DataFrame(columns=colums, data=values_T)
    f1 = open(csv_path, mode='w', newline='')
    save.to_csv(f1, encoding='gbk', index=False)
    f1.close()


def read_data(csv_path):
    pd_data = pd.read_csv(csv_path, sep=',', header='infer', usecols=['Value'])
    # pd_data['Status'] = pd_data['Status'].values
    return pd_data
