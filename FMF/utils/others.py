import datetime
import os


def get_acc_num(output, labels):
    """
    :param output: (N,C)
    :param labels: (N,)
    :return: double
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum()


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def make_dir(file):
    if not os.path.exists(file):
        os.makedirs(file)


