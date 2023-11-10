import numpy as np
import pandas as pd
import os
from os.path import join as ospj
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

color_par = {
    'cnn': '#5D9A6B',
    'widecnn': '#B55D60',
    'rnn': '#5875A4',
    'std': '#857AAB'
}

marker_par = {
    'cnn': '.',
    'widecnn': 'o',
    'rnn': 'v'
}


def read_data(csv_path):
    pd_data = pd.read_csv(csv_path, sep=',', header='infer', usecols=['Step', 'Value'])
    # pd_data['Status'] = pd_data['Status'].values
    return pd_data

def draw_graph(metric_data, img_path):
    x_axis_data = np.linspace(1, 5, 5)
    for i, k in enumerate(metric_data.keys()):
        if k == 'name':
            continue
        plt.plot(x_axis_data, metric_data[k], color=color_par[k], marker=marker_par[k], alpha=1, linewidth=1,
                 label=k)
    plt.legend()  # 显示图例
    plt.grid(ls='--')  # 生成网格
    plt.xlabel('epoch')
    plt.ylabel(metric_data['name'])
    # plt.title('CIFAR-10 M5, Boundary Attack, Threshold Choosing')
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    # plt.ylim(0.5, 1.05)

    # plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.savefig(img_path)
    plt.clf()


if __name__ == '__main__':
    log_dir = 'log'
    image_dir = 'image'
    model_list = ['cnn', 'widecnn', 'rnn']
    metric_list = ['train_acc', 'train_loss', 'valid_loss']

    for i, me in enumerate(metric_list):
        me_data = {}
        me_data['name'] = me
        img_path = ospj(image_dir, 'b16_' + me + '.png')
        for j, mo in enumerate(model_list):
            csv_path = ospj(log_dir, mo, 'b16_' + me + '.csv')
            data_pd = read_data(csv_path)
            me_data[mo] = data_pd['Value'].values
        draw_graph(me_data, img_path)

