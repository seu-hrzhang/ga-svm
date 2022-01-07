from numpy import *
import numpy as np
import random
import matplotlib.pyplot as plt


def random_var(i, m):
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

        # record of error vector
        self.e = np.zeros(self.m, dtype=float)
        self.get_error()

        # outer and inner variables to optimize
        self.outer_var = -1
        self.inner_var = -1
        self.mark = -1

        # initialize 'k'
        for i in range(self.m):
            for j in range(self.m):
                self.k[i, j] = self.x[i, :] * self.x[j, :].T

    def g(self, i):
        return dot(self.alpha * self.y, self.k[:, i]) + self.b
        # return float(multiply(self.alpha, self.y).T * (self.x * self.x[i, :].T)) + self.b

    def get_error(self):
        self.e = [self.g(i) - float(self.y[i]) for i in range(self.m)]

    def update(self, i, j):
        self.get_error()

        # k_12 = k_21
        k_ii = self.k[i, i]
        k_ij = self.k[i, j]
        k_jj = self.k[j, j]
        eta = k_ii + k_jj - 2.0 * k_ij

        if eta <= 0:
            print('eta <= 0')
            return 0

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

        # if l_lim == h_lim:
        #     return 0

        # alpha[j] > l_lim, alpha[j] < h_lim
        alpha_jn = max(min((alpha_jo + self.y[j] * (self.e[i] - self.e[j]) / eta), h_lim), l_lim)
        self.alpha[j] = alpha_jn

        # ensure enough change in 'alpha[j]'
        if abs(alpha_jn - alpha_jo) < 0.00001:
            return 0

        # calc 'alpha[i]' using 'alpha[j]'
        alpha_in = alpha_io + self.y[i] * self.y[j] * (alpha_jo - alpha_jn)
        self.alpha[i] = alpha_in

        self.get_error()

        # update thresholds 'b'
        b_i = self.b - self.e[i] - self.y[i] * k_ii * (alpha_in - alpha_io) - self.y[j] * k_ij * (alpha_jn - alpha_jo)
        b_j = self.b - self.e[j] - self.y[i] * k_ij * (alpha_in - alpha_io) - self.y[j] * k_jj * (alpha_jn - alpha_jo)

        if 0 < alpha_in < self.const:
            self.b = b_i
        elif 0 < alpha_jn < self.const:
            self.b = b_j
        else:
            self.b = (b_i + b_j) / 2.0
        return 1

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
        valid_indices = [i for i, alpha in enumerate(self.alpha) if 0 < alpha < self.const]

        if len(valid_indices) > 1:
            inner_var = -1
            max_delta = 0
            for i in valid_indices:
                if i == self.outer_var:
                    continue
                delta = abs(self.e[self.outer_var] - self.e[inner_var])
                if delta > max_delta:
                    inner_var = i
                    max_delta = delta
        else:
            inner_var = random_var(self.outer_var, self.m)

        self.inner_var = inner_var
        return self.inner_var

    def verify_outer_var(self):
        i = self.outer_var
        r = self.y[i] * self.e[i]
        if r < -self.err and self.alpha[i] < self.const or r > self.err and self.alpha[i] > 0:
            self.get_inner_var()
            return self.update(self.outer_var, self.inner_var)
        else:
            return 0

    def get_w(self):
        for i in range(self.m):
            self.w += self.x[i, :] * float(self.alpha[i] * self.y[i])
        self.w = self.w[0]
        return self.w

    def naive_smo(self):
        while self.it <= self.max_it:
            changes = 0
            for i in range(self.m):
                self.e[i] = self.g(i) - float(self.y[i])
                if self.y[i] * self.e[i] < -self.err and self.alpha[i] < self.const \
                        or self.y[i] * self.e[i] > self.err and self.alpha[i] > 0:
                    j = random_var(i, self.m)
                    self.e[j] = self.g(j) - float(self.y[j])
                    if self.update(i, j) == 1:
                        changes += 1
                    else:
                        continue
            if changes == 0:
                self.it += 1
            else:
                self.it = 0
        return True

    def platt_smo(self):
        entire = True
        while self.it <= self.max_it:
            changes = 0
            if entire:
                for i in range(self.m):
                    self.outer_var = i
                    changes += self.verify_outer_var()
            else:
                non_bound_indices = [i for i, alpha in enumerate(self.alpha) if 0 < alpha < self.const]
                for i in non_bound_indices:
                    self.outer_var = i
                    changes += self.verify_outer_var()
            self.it += 1

            if entire:
                entire = False
            elif changes == 0:
                entire = True

    def plot(self, title):
        # scatter source data
        for i in range(100):
            if self.alpha[i] > 0:
                if self.labels[i] != 1:
                    plt.scatter(self.data[i][0], self.data[i][1], color='red')
                else:
                    plt.scatter(self.data[i][0], self.data[i][1], color='green')

        self.get_w()

        # print params
        print('w: ', self.w)
        print('b: ', self.b)

        k = -self.w[0] / self.w[1]
        bias = -self.b / self.w[1]
        print('k: ', k)
        print('bias: ', bias)

        # plot hyperplane
        x = np.linspace(-1, 10, 100)
        y = (-self.b - self.w[0] * x) / self.w[1]
        plt.plot(x, y)

        plt.axis([-1, 10, -6, 5])
        plt.title(title)
        plt.show()
