import math
import random
import matplotlib.pyplot as plt
import numpy as np


def fitness(x, y, indiv):
    """
    get fitness of a given individual based on standard form of SVM optimization
    :param x: coordinates of data
    :param y: label of data
    :param indiv: given GA individual
    :return: fitness of given individual
    """
    w = indiv[0:2]
    b = indiv[2]
    return ((w[0] * x[0] + w[1] * x[1] + b) * y) / math.sqrt(w[0] ** 2 + w[1] ** 2)


class GA:
    def __init__(self, data, labels, pop_size, n_params, max_it, cross_rate=0.8, mutation=0.1):
        # input data
        self.data = data
        self.labels = labels

        # GA params
        self.pop_size = pop_size  # size of population
        self.n_params = n_params  # param number
        self.max_it = max_it  # maximum iterations
        self.cross_rate = cross_rate  # crossover rate
        self.mutation = mutation  # mutation rate

        self.cross_point = 0
        self.mutate_point = 0

        # record of params
        self.w = []
        self.b = []

        self.pop = [[random.random() * 4 - 2 for i in range(n_params)] for j in range(pop_size)]

    def crossover(self, child_1, child_2):
        rand = random.random()
        if rand <= self.cross_rate:
            self.cross_point = int(round(random.uniform(1, self.n_params - 1)))
            for i in range(self.cross_point):
                child_1[i], child_2[i] = child_2[i], child_1[i]

    def mutate(self, child):
        rand = random.random()
        if rand <= self.mutation:
            self.mutate_point = int(round(random.uniform(0, self.n_params - 1)))
            child[self.mutate_point] += random.random() * 4 - 2

    def select(self):
        # get fitness of each individual
        # fitness defined as the minimum fitness of all given data
        fit = [min([fitness(self.data[j], self.labels[j], self.pop[i]) for j in range(len(self.data))]) for i in
               range(len(self.pop))]

        next_gen = sorted(range(len(self.pop)), key=lambda k: fit[k], reverse=True)
        self.pop = [self.pop[next_gen[i]] for i in range(self.pop_size)]

    def record_params(self):
        self.w.append(self.pop[0][0:2])
        self.b.append(self.pop[0][2])

    def evolve(self):
        it = 0
        while it < self.max_it:
            children = []

            # select parents
            parent_1 = self.pop[int(round(random.uniform(0, self.n_params)))][:]
            parent_2 = self.pop[int(round(random.uniform(0, self.n_params)))][:]

            # producing children
            child_1 = parent_1.copy()
            child_2 = parent_2.copy()

            # crossover
            self.crossover(child_1, child_2)

            # mutation
            self.mutate(child_1)
            self.mutate(child_2)

            # add children to population
            children.append(child_1)
            children.append(child_2)
            self.pop += children

            # selection
            self.select()

            it += 1
            self.record_params()

    def plot(self, title):
        w = self.pop[0][0:2]
        b = self.pop[0][2]

        k = [-self.w[i][0] / self.w[i][1] for i in range(len(self.w))]
        bias = [-self.b[i] / self.w[i][1] for i in range(len(self.b))]

        # plot history of params
        x = np.linspace(0, 1000, 1000)
        plt.xlabel('Generations')
        plt.ylabel('Slope')
        plt.plot(x, k, color='tab:blue')
        plt.show()

        plt.xlabel('Generations')
        plt.ylabel('Bias')
        plt.plot(x, bias, color='tab:orange')
        plt.show()

        for i in range(len(self.data)):
            if self.labels[i] == 1:
                plt.scatter(self.data[i][0], self.data[i][1], color='green')
            else:
                plt.scatter(self.data[i][0], self.data[i][1], color='red')

        x = np.linspace(-1, 10, 100)
        y = (-b - w[0] * x) / w[1]
        plt.plot(x, y)
        plt.axis([-1, 10, -6, 5])
        plt.title(title)
        plt.show()
