import matplotlib.pyplot as plt
from math import sqrt
import threading
import numpy as np
from sklearn.utils import shuffle


def predict(x, coefficients):
    y_predicted = coefficients[0]
    for i in range(len(x)):
        y_predicted += x[i] * coefficients[i+1]

    return y_predicted


def sgd(train, alpha, epochs):
    error_history = []
    w_ = [1.0 for _ in range(len(train[0]))]
    x_ = [row[:-1] for row in list(train)]
    y_ = [row[-1] for row in train]

    for epoch in range(epochs):
        # x_, y_ = shuffle(x_, y_)

        epoch_error = []
        for x__, y__ in zip(x_, y_):
            error = y__ - predict(x__, w_)
            epoch_error.append(error**2)
            w_[0] += (alpha*error)
            for i in range(len(x__)):
                w_[i + 1] += (alpha * error * x__[i])

        error_history.append(sum(epoch_error)/len(epoch_error))
    return w_, error_history

if __name__ == '__main__':
    alpha = 0.001
    epochs = 1000
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')

    x = data[:, 0]
    y = data[:, 1]
    # for i in x:
    #     print(i)
    plt.plot(x, y, "ko")
    p0 = [0, 0]
    p1 = [-3.8465808694666443, 1.1507969424916813]
    plt.plot(0, 0, -3.8465808694666443, 1.1507969424916813, 'ro')
    plt.show()
    # plt.savefig('results/plot')
    w, epoch_errors = sgd(train=data, alpha=alpha, epochs=epochs)
    # print(sum(epoch_errors)/len(epoch_errors))
    # print(w)
    # plt.plot([i for i in range(0, epochs)], epoch_errors)
    # plt.show()


    # data = np.loadtxt("data/ex1data2.txt")
    # w, epoch_errors = sgd(train=data, alpha=alpha, epochs=epochs)
    # print(w)
    #
    # plt.plot([i for i in range(0, epochs)], epoch_errors)
    # plt.show()
