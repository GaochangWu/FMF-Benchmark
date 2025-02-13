import os
import matplotlib.pyplot as plt


def SubPlot(metrix, xlabel='epoch', ylabel='loss'):
    train_m = metrix[0]
    test_m = metrix[1]
    x = list(range(1, len(train_m)+1))
    plt.plot(x, train_m, label='train')
    plt.plot(x, test_m, label='test')
    plt.legend(loc='best')  # 展示图例
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot2(metrics, xlabel='epoch', ylabels=[], out_dir='', titles=None):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    SubPlot(metrics[0], xlabel, ylabels[0])
    plt.title(ylabels[0].upper() if titles is None else titles[0])

    plt.subplot(1, 2, 2)
    SubPlot(metrics[1], xlabel, ylabels[1])
    plt.title(ylabels[1].upper() if titles is None else titles[1])

    plt.savefig(os.path.join(out_dir, 'plot_two_metrics.png'))


def plot3(metrics, xlabel='epoch', ylabels=[], out_dir='', titles=None):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    SubPlot(metrics[0], xlabel, ylabels[0])
    plt.title(ylabels[0].upper() if titles is None else titles[0])

    plt.subplot(1, 3, 2)
    SubPlot(metrics[1], xlabel, ylabels[1])
    plt.title(ylabels[1].upper() if titles is None else titles[1])

    plt.subplot(1, 3, 3)
    SubPlot(metrics[2], xlabel, ylabels[2])
    plt.title(ylabels[2].upper() if titles is None else titles[2])

    plt.savefig(os.path.join(out_dir, 'plot_three_metrics.png'))
