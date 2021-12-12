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


def get_visual(data, labels, title='Figure'):
    """
    visualize input data and labels using matplotlib
    :param data: input data
    :param labels: input labels
    :param title: figure title
    :return: none
    """
    data = np.array(data)
    labels = np.array(labels)
    rows = data.shape[0]

    x = data[:, 0]
    y = data[:, 1]

    for i in range(rows):
        if labels[i] == 1:
            plt.scatter(x[i], y[i], color='green')
        else:
            plt.scatter(x[i], y[i], color='red')
    plt.axis([-1, 10, -6, 5])
    plt.title(title)
    plt.show()


def run():
    data, labels = load_data('data.txt')
    get_visual(data, labels, 'Input Data')

    smo_solver = SMO(data, labels, 0.6, 0.001, max_it=40)
    if smo_solver.train_naive():
        smo_solver.plot()
        print(smo_solver.w)
        print(smo_solver.b)
    else:
        print('SMO process failed')


if __name__ == '__main__':
    run()
