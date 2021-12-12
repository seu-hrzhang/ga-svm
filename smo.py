from numpy import *
import numpy as np
import random
import matplotlib.pyplot as plt


def get_random_id(i, m):
    """
    randomly select variable index 'j'
    :param i: given index 'i' to distinguish from
    :param m: upper limit of index
    :return: mark index 'j'
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def smo_naive(data, labels, const, toler, max_iter):
    """
    Naive implemantation of SMO algo. Target variables mark randomly.
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

                # k_12 = k_21
                k_11 = x[i, :] * x[i, :].T
                k_12 = x[i, :] * x[j, :].T
                k_22 = x[j, :] * x[j, :].T
                eta = k_11 + k_22 - 2.0 * k_12

                # alpha[j] > l_lim, alpha[j] < h_lim
                alpha_jn = max(min((alpha_jo + y[j] * (e_i - e_j) / eta), h_lim), l_lim)  # restrict range of 'alpha[j]'
                alpha[j] = alpha_jn

                if abs(alpha_jn - alpha_jo < 0.00001):
                    # print('little change in alpha[j]')
                    continue

                alpha_in = alpha_io + y[j] * y[i] * (alpha_jo - alpha_jn)
                alpha[i] = alpha_in

                b_i = b - e_i - y[i] * k_11 * (alpha_in - alpha_io) - y[j] * k_12 * (alpha_jn - alpha_jo)
                b_j = b - e_j - y[i] * k_12 * (alpha_in - alpha_io) - y[j] * k_22 * (alpha_jn - alpha_jo)

                # if 0 < alpha_in < const:
                #     b = b_i
                # elif 0 < alpha_jn < const:
                #     b = b_j
                # else:
                b = (b_i + b_j) / 2.0

                changes += 1
                # print('iter: %d, i=%d, changes=%d' % (it, i, changes))

        if changes == 0:
            it += 1
        else:
            it = 0

    return alpha, b


class SMO:
    def __init__(self, data, labels, const, err, max_it):
        # input data
        self.data = data
        self.labels = labels
        self.x = mat(data)
        self.y = array(labels).transpose()
        self.m, self.n = shape(self.x)

        # SMO params
        self.const = const
        self.err = err
        self.max_it = max_it
        self.it = 0  # number of iterations

        # SVM params
        self.alpha = array(zeros(self.m), dtype='float64')
        self.b = 0.0
        self.w = np.zeros([1, self.n], dtype=float)

        # pre-calculated 'k' matrix
        self.k = zeros((self.m, self.m), dtype='float64')

        # record of 'e' vector
        self.e = np.zeros(self.m, dtype=float)

        # outer and inner variables to optimize
        self.outer_var = -1
        self.inner_var = -1
        self.mark = -1

        # initialize 'k' and 'e'
        for i in range(self.m):
            for j in range(self.m):
                self.k[i, j] = self.x[i, :] * self.x[j, :].T

        for i in range(self.m):
            self.e[i] = self.g(i) - float(self.y[i])

    def g(self, i):
        return dot(self.alpha * self.y, self.k[:, i]) + self.b
        # return float(multiply(self.alpha, self.y).T * (self.x * self.x[i, :].T)) + self.b

    def update(self, i, j):
        # record old values of 'alpha_i' and 'alpha_j'
        alpha_io = self.alpha[i].copy()
        alpha_jo = self.alpha[j].copy()

        # get upper/lower limits
        if self.y[i] != self.y[j]:
            l_lim = max(0, alpha_jo - alpha_io)
            h_lim = min(self.const, self.const + alpha_jo - alpha_io)
        else:
            l_lim = max(0, alpha_io + alpha_jo - self.const)
            h_lim = min(self.const, alpha_io + alpha_jo)

        if l_lim == h_lim:
            return -1

        # k_12 = k_21
        k_11 = self.k[i, i]
        k_12 = self.k[i, j]
        k_22 = self.k[j, j]
        eta = k_11 + k_22 - 2.0 * k_12

        # alpha[j] > l_lim, alpha[j] < h_lim
        alpha_jn = max(min((alpha_jo + self.y[j] * (self.e[i] - self.e[j]) / eta), h_lim), l_lim)
        self.alpha[j] = alpha_jn

        # ensure enough change in 'alpha[j]'
        if abs(alpha_jn - alpha_jo) < 0.00001:
            return -2

        # calc 'alpha[i]' using 'alpha[j]'
        alpha_in = alpha_io + self.y[j] * self.y[i] * (alpha_jo - alpha_jn)
        self.alpha[i] = alpha_in

        # update param 'b'
        b_i = self.b - self.e[i] - self.y[i] * k_11 * (alpha_in - alpha_io) - self.y[j] * k_12 * (alpha_jn - alpha_jo)
        b_j = self.b - self.e[j] - self.y[i] * k_12 * (alpha_in - alpha_io) - self.y[j] * k_22 * (alpha_jn - alpha_jo)
        self.b = (b_i + b_j) / 2.0

        # update param 'e'
        # self.e[i] = self.g(i) - float(self.y[i])
        # self.e[j] = self.g(j) - float(self.y[j])

        return 0

    def get_outer_var(self):
        # condition 1: 0 < alpha[i] < C and y[i] * g_xi == 1
        for i in range(self.m):
            if 0 < self.alpha[i] < self.const and abs(self.y[i] * self.e[i]) > self.err and i > self.mark:
                self.outer_var = i
                return self.outer_var

        # condition 2: alpha[i] == 0 and y[i] * g_xi >= 1
        for i in range(self.m):
            if self.alpha[i] == 0 and self.y[i] * self.g(i) < 1 and i > self.mark:
                self.outer_var = i
                return self.outer_var

        # condition 3: alpha[i] == C and y[i] * g_xi <= 1
        for i in range(self.m):
            if self.alpha[i] == self.const and self.y[i] * self.g(i) > 1 and i > self.mark:
                self.outer_var = i
                return self.outer_var
        return -1

    def get_inner_var(self):
        if self.outer_var == -1:
            return -1

        m, n = shape(self.x)

        if self.e[self.outer_var] > 0:
            inner_idx = where(self.e == np.min(self.e))
        else:
            inner_idx = where(self.e == np.max(self.e))

        self.inner_var = inner_idx[0][int(random.uniform(0, len(inner_idx)))]
        return self.inner_var

    def get_surface(self):
        for i in range(self.m):
            self.w += self.x[i, :] * float(self.alpha[i] * self.y[i])
        self.w = self.w[0]
        return self.w, self.b

    def plot(self):
        # scatter source data
        for i in range(100):
            if self.alpha[i] > 0:
                if self.labels[i] == 1:
                    plt.scatter(self.data[i][0], self.data[i][1], color='green')
                else:
                    plt.scatter(self.data[i][0], self.data[i][1], color='red')
        # plot classification surface
        self.get_surface()
        x = np.linspace(-2, 10, 100)
        y = (self.b - self.w[1] * x) / self.w[0]
        plt.plot(x, y)

        plt.axis([-1, 10, -6, 5])
        plt.title('Support Vectors')
        plt.show()

    # check loop conditions
    def loop(self):
        # condition 1: sum of alpha[i] * y[i] == 0
        # sum = 0.0
        # for i in range(self.m):
        #     sum += self.alpha[i] * self.y[i]
        # if abs(sum) > self.err:
        #     return True

        # condition 2: y[i] * g[i] == 1 and 0 < alpha[i] < C
        for i in range(self.m):
            if abs(self.y[i] * self.e[i]) > self.err or self.alpha[i] < 0 or self.alpha[i] > self.const:
                print('elem ' + str(i) + ' breaks restraint')
                return True
        return False

    def run_naive(self):
        while self.it <= self.max_it:
            print('iter: ', self.it)
            changes = 0
            for i in range(self.m):
                self.e[i] = self.g(i) - float(self.y[i])
                if self.y[i] * self.e[i] < -self.err and self.alpha[i] < self.const \
                        or self.y[i] * self.e[i] > self.err and self.alpha[i] > 0:
                    j = get_random_id(i, self.m)
                    self.e[j] = self.g(j) - float(self.y[j])
                    if self.update(i, j) == 0:
                        changes += 1
                    else:
                        continue
            if changes == 0:
                self.it += 1
            else:
                self.it = 0
        return True

    def run(self):
        while self.loop():
            print('iter: %d' % self.it)

            # check of iterations
            self.it += 1
            if self.it >= self.max_it:
                print('reaching maximum iterations')
                return True

            if self.get_outer_var() == -1:
                print('error getting outer variable')
                self.mark = -1
                continue
            else:
                print('outer variable: ', self.outer_var)

            if self.get_inner_var() == -1:
                print('error getting inner variable')
                return False
            else:
                print('inner variable: ', self.inner_var)

            rt_val = self.update(self.outer_var, self.inner_var)
            print('returning ', rt_val)
            if rt_val != 0:
                self.mark = self.outer_var
                continue
            else:
                self.mark = -1

        return True
