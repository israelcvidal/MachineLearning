import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import matplotlib
import pylab
import random


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def logistic_predict(x, coefficients):
    y_predicted = coefficients[0]
    for i in range(len(x)):
        y_predicted += x[i] * coefficients[i+1]

    return sigmoid(y_predicted)


def sgd(train_, alpha_, epochs_):
    error_history = []
    w_ = [random.random()*20 for _ in range(len(train_[0]))]

    for epoch in range(epochs_):
        train_ = shuffle(train_)

        epoch_error = []
        for row in train_:
            error = row[-1] - logistic_predict(row[:-1], w_)
            epoch_error.append(error**2)
            w_[0] += (alpha_ * error)
            for i in range(len(row) - 1):
                w_[i + 1] += (alpha_ * error * row[i])

        error_history.append(sum(epoch_error)/(2*len(epoch_error)))
    return w_, error_history


def least_squares(X, y):
    # calculate the least square
    shape_ = np.matrix(X).shape
    x_ = np.ones((shape_[0], shape_[1]+1))
    x_[:, 1:] = X
    y_ = np.matrix(y)
    return np.linalg.inv(x_.transpose().dot(x_)).dot(x_.transpose()).dot(y_.transpose())


def regularized_least_squares(X, y, lambda_):
    # calculate the least square
    shape_ = np.matrix(X).shape
    x_ = np.ones((shape_[0], shape_[1] + 1))
    x_[:, 1:] = X
    y_ = np.matrix(y)
    lambda_identity = lambda_ * np.identity(shape_[1] + 1)
    lambda_identity[0, 0] = 0
    return np.linalg.inv(x_.transpose().dot(x_) + lambda_identity).dot(x_.transpose()).dot(y_.transpose())


def mse(real, predicted):
    error = 0
    for i, j in zip(real, predicted):
        error += (i-j)**2
    return error/(2*len(real))

if __name__ == '__main__':
    data_1 = np.loadtxt("data/ex2data1.txt", delimiter=",")
    normalize(data_1[:-1], axis=0, copy=False)
    train = data_1[:int(len(data_1)*0.7)]
    test = data_1[int(len(data_1)*0.7):]
    alpha = 0.01
    epochs = 1000

    # # plot dataset1:
    # x = data_1[:, :-1]
    # y = data_1[:, -1]
    # colors = ['#1b9e77', '#d95f02']
    # plt.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors), s=50)
    # # plt.show()
    #
    # # save plot1:
    # plt.savefig('results/plot_data_1')
    # plt.close()




    # # run logistic refression for dataset1:
    # w, epoch_errors = sgd(train_=data_1, alpha_=alpha, epochs_=epochs)
    #
    # # show results for dataset1:
    # print("w: " + str(w))
    # plt.plot([i for i in range(0, epochs)], epoch_errors)
    # # # plt.show()
    # #
    # # # save results for dataset1
    # plt.savefig('results/epochs_mse_dataset1')
    # plt.close()
    #
    # predicted_test = [logistic_predict(x, w) for x in test[:, :-1]]
    # test_error = mse(test[:, -1], predicted_test)
    # print("test error: ", test_error)







    #################################################



