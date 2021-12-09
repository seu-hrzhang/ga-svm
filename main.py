import numpy as np
import matplotlib.pyplot as plt

from smo import *


def load_data(filename):
    """
    load data from file
    :param filename: path of data file
    :return: data and label matrices
    """
    file = open(filename)
    data = []
    labels = []
    for line in file.readlines():
        line_attr = line.strip().split('\t')
        data.append([float(x) for x in line_attr[:-1]])
        labels.append(float(line_attr[-1]))
    return data, labels


def get_visial(data, labels):
    data = np.array(data)
    labels = np.array(labels)
    [rows, cols] = data.shape

    x = data[:, 0]
    y = data[:, 1]

    for i in range(rows):
        if labels[i] == 1:
            plt.scatter(x[i], y[i], color='green')
        else:
            plt.scatter(x[i], y[i], color='red')
    plt.axis([-1, 10, -6, 5])
    plt.show()


def run():
    data, labels = load_data('data.txt')
    # get_visial(data, labels)

    alpha, b = smo_naive(data, labels, 0.6, 0.001, 40)

    # print(b)
    # print(alpha)

    # for i in range(100):
    #     if alpha[i] > 0:
    #         if labels[i] == 1:
    #             plt.scatter(data[i][0], data[i][1], color='green')
    #         else:
    #             plt.scatter(data[i][0], data[i][1], color='red')
    # plt.axis([-1, 10, -6, 5])
    # plt.show()

    # TODO: get classification surface


if __name__ == '__main__':
    run()
