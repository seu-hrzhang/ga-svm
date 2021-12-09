import numpy as np
from numpy import *
import sys


def get_random_id(i, m):
    """
    randomly select variable index 'j'
    :param i: given index 'i' to distinguish from
    :param m: upper limit of index
    :return: selected index 'j'
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def smo_naive(data, labels, const, toler, max_iter):
    """
    Naive implemantation of SMO algo. Target variables selected randomly.
    :param data: input data (from 'load_data'), aka variable x
    :param labels: labels of data (from 'load_data'), aka variable y
    :param const: constant limit of parameter alpha
    :param toler:
    :param max_iter: Maximum iterations
    :return:
    """
    # 'x' and 'y' both columm matrix/vector
    x = mat(data)
    y = mat(labels).transpose()

    b = 0.0
    it = 0  # count of iterations
    m, n = shape(x)
    alpha = mat(zeros((m, 1)))

    while it < max_iter:
        changes = 0
        for i in range(m):
            g_xi = float(multiply(alpha, y).T * (x * x[i, :].T)) + b
            e_i = g_xi - float(y[i])

            if y[i] * e_i < -toler and alpha[i] < const or y[i] * e_i > toler and alpha[i] > 0:
                j = get_random_id(i, m)
                g_xj = float(multiply(alpha, y).T * (x * x[j, :].T)) + b
                e_j = g_xj - float(y[j])

                # record old values of 'alpha_i' and 'alpha_j'
                alpha_io = alpha[i].copy()
                alpha_jo = alpha[j].copy()

                if y[i] != y[j]:
                    l_lim = max(0, alpha_jo - alpha_io)
                    h_lim = min(const, const + alpha_jo - alpha_io)
                else:
                    l_lim = max(0, alpha_io + alpha_jo - const)
                    h_lim = min(const, alpha_io + alpha_jo)

                if l_lim == h_lim:
                    # print('limits overlapped')
                    continue

                k_11 = x[i, :] * x[i, :].T
                k_12 = x[i, :] * x[j, :].T
                k_22 = x[j, :] * x[j, :].T

                eta = 2.0 * k_12 - k_11 - k_22

                if eta >= 0:
                    # print('eta larger than 0')
                    continue

                # alpha[j] > l_lim, alpha[j] < h_lim
                alpha_jn = max(min((alpha_jo - y[j] * (e_i - e_j) / eta), h_lim), l_lim)  # restrict range of 'alpha[j]'
                alpha[j] = alpha_jn

                if abs(alpha_jn - alpha_jo < 0.00001):
                    # print('little change in alpha[j]')
                    continue

                alpha_in = alpha_io + y[j] * y[i] * (alpha_jo - alpha_jn)
                alpha[i] = alpha_in

                b_i = b - e_i - y[i] * k_11 * (alpha_in - alpha_io) - y[j] * k_12 * (alpha_jn - alpha_jo)
                b_j = b - e_j - y[i] * k_12 * (alpha_in - alpha_io) - y[j] * k_22 * (alpha_jn - alpha_jo)

                if 0 < alpha_in < const:
                    b = b_i
                elif 0 < alpha_jn < const:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2.0

                changes += 1
                # print('iter: %d, i=%d, changes=%d' % (it, i, changes))

        if changes == 0:
            it += 1
        else:
            it = 0
        # print('%d iterations' % it)
    return alpha, b
