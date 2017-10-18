import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import matplotlib
import pylab
import random


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def logistic_predict(x, coefficients, label=False):
    y_predicted = coefficients[0]
    for i in range(len(x)):
        y_predicted += x[i] * coefficients[i+1]
    if label:
        predicted = sigmoid(y_predicted)
        if predicted >= 0.5:
            return 1
        else:
            return 0
    return sigmoid(y_predicted)


def sgd(train_, alpha_, epochs_):
    error_history = []
    # w_ = [random.random() for _ in range(len(train_[0]))]
    w_ = [0 for _ in range(len(train_[0]))]

    for epoch in range(epochs_):
        train_ = shuffle(train_)

        epoch_error = []
        for row in train_:
            error = row[-1] - logistic_predict(row[:-1], w_)
            epoch_error.append(error**2)
            w_[0] += (alpha_ * error)
            for i in range(len(row) - 1):
                w_[i + 1] += (alpha_ * error * row[i])

        error_history.append(sum(epoch_error)/len(epoch_error))
    return w_, error_history


def least_squares(X, y):
    # calculate the least square
    shape_ = np.matrix(X).shape
    x_ = np.ones((shape_[0], shape_[1]+1))
    x_[:, 1:] = X
    y_ = np.matrix(y)
    return np.linalg.inv(x_.transpose().dot(x_)).dot(x_.transpose()).dot(y_.transpose())


def sgd_regularized(train_, alpha_, epochs_, lambda_):
    error_history = []
    w_ = [0 for _ in range(len(train_[0]))]

    for epoch in range(epochs_):
        train_ = shuffle(train_)

        epoch_error = []
        for row in train_:
            error = row[-1] - logistic_predict(row[:-1], w_)
            epoch_error.append(error**2)
            w_[0] += (alpha_ * error * row[0])
            for i in range(len(row) - 1):
                w_[i + 1] += alpha_ * ((error * row[i]) - lambda_*w_[i+1])

        error_history.append(sum(epoch_error)/len(epoch_error))
    return w_, error_history


def mse(real, predicted):
    error_ = 0
    for i, j in zip(real, predicted):
        error_ += (i-j)**2
    return error_/len(real)


def k_fold(data_, alpha_, epochs_, k):
    fold_size = int(len(data_)/k)

    folds = [data_[i*fold_size: (i+1)*fold_size, :] for i in range(k)]
    ws_ = []
    error_ = []
    for i in range(len(folds)):
        test_ = folds[i]
        train_ = []
        for k in np.array([x for j, x in enumerate(folds) if j != i]):
            train_.extend(k)

        w_, _ = sgd(train_, alpha_, epochs_)
        predicted_test = [logistic_predict(x, w_, label=True) for x in test_[:, :-1]]
        test_error = sum(abs(test_[:, -1] - predicted_test))/len(predicted_test)
        ws_.append(w_)
        error_.append(test_error)

    return ws_, sum(error_)/len(error_)


def plot_data(data_, save=False):
    x = data_[:, :-1]
    y = data_[:, -1]
    colors = ['#1b9e77', '#d95f02']
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors), s=30)
    if save:
        plt.savefig('results/' + save)
    else:
        plt.show()

    plt.close()


def plot_epochs_error(epochs_, epoch_errors, save=None):
    plt.plot([i for i in range(0, epochs_)], epoch_errors)
    if save:
        plt.savefig('results/epochs_mse_dataset1')
    else:
        plt.show()
    plt.close()


def question_1():
    data_1 = np.loadtxt("data/ex2data1.txt", delimiter=",")

    # # plot dataset1:
    # plot_data(data_1)

    normalize(data_1[:, :-1], axis=0, copy=False)
    train = data_1[:int(len(data_1) * 0.7), :]
    test = data_1[int(len(data_1) * 0.7):, :]
    alpha = 0.01
    epochs = 1000

    # # run logistic refression for dataset1:
    w, epoch_errors = sgd(train_=train, alpha_=alpha, epochs_=epochs)

    # show results for dataset1:
    print("w: " + str(w))
    plot_epochs_error(epochs, epoch_errors, save="epochs_mse_dataset1")
    predicted_test = [logistic_predict(x, w, label=True) for x in test[:, :-1]]
    test_error = sum(abs(test[:, -1] - predicted_test))
    test_error_percent = test_error / len(predicted_test)
    print("test error: ", test_error, test_error_percent)

    print("\nKFOLD:")
    ws, error = k_fold(data_1, alpha_=alpha, epochs_=epochs, k=5)
    for i, w in enumerate(ws):
        print("w" + str(i) + ": ", w)
    print(error)


def question_2():
    data_2 = np.loadtxt("data/dataset_mapfeature.txt", delimiter=",")
    # plot_data(data_2)

    normalize(data_2[:, :-1], axis=0, copy=False)
    alpha = 0.01
    epochs = 1000
    for lambda_ in [0, 0.01, 0.25]:
        W, _ = sgd_regularized(data_2[:, :-1], alpha, epochs, lambda_)
        print(W)


def main():
    question_1()
    question_2()

    #################################################


if __name__ == '__main__':
    main()