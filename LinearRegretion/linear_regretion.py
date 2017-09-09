import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def predict(x, coefficients):
    y_predicted = coefficients[0]
    for i in range(len(x)):
        y_predicted += x[i] * coefficients[i+1]

    return y_predicted


def sgd(train_, alpha_, epochs_):
    error_history = []
    w_ = [1.0 for _ in range(len(train_[0]))]

    for epoch in range(epochs_):
        train_ = shuffle(train_)

        epoch_error = []
        for row in train_:
            error = row[-1] - predict(row[:-1], w_)
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
    # alpha = 0.001
    # epochs = 1000
    # dataset1:
    # data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    #
    # # plot dataset1:
    # x = data[:, 0]
    # y = data[:, 1]
    # line = [predict([i], [-3.917517058471562, 1.303534229171897]) for i in x]
    # plt.plot(x, y, "kx")
    # plt.plot(x, line, "r-")
    # # plt.show()
    #
    # # save plot1:
    # # plt.savefig('results/plot_data')
    # plt.savefig('results/plot_data_line')

    # run sgd for dataset1:
    # w, epoch_errors = sgd(train=data, alpha=alpha, epochs=epochs)

    # # show results for dataset1:
    # print("w: " + str(w))
    # plt.plot([i for i in range(0, epochs)], epoch_errors, "k-")
    # # plt.show()
    #
    # #save results for dataset1
    # plt.savefig('results/epochs_mse_dataset1')


    #################################################


    # # dataset2:
    # alpha = 0.01
    # epochs = 100
    #
    # data2 = np.loadtxt("data/ex1data2.txt")
    #
    # w, epoch_errors = sgd(train=data2, alpha=alpha, epochs=epochs)
    #
    # # show results for dataset2:
    # print("w: " + str(w))
    # plt.plot([i for i in range(0, epochs)], epoch_errors, 'k-')
    # plt.show()
    #
    # # save results for dataset2
    # # plt.savefig('results/epochs_mse_dataset2')
    #
    # # X = data2[:, :-1]
    # # y = data2[:, -1]
    # #
    # # w = least_squares(X, y)
    # # print(w)

    ################################################

    # dataset3:
    alpha = 0.01
    epochs = 100

    data3 = np.loadtxt("data/ex1data3.txt")

    train = data3[:30]
    test = data3[30:]

    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = test[:, :-1]
    y_test = test[:, -1]

    mse_train = []
    mse_test = []

    for lambda_ in range(6):
        W = regularized_least_squares(x_train, y_train, lambda_).tolist()
        W = [w[0] for w in W]

        # train
        predicted_train = [predict(x, W) for x in x_train]

        # test
        predicted_test = [predict(x, W) for x in x_test]

        # mean squared error
        mse_train.append(mse(y_train, predicted_train))
        mse_test.append(mse(y_test, predicted_test))

    # plt.plot([i for i in range(6)], mse_train, 'ko')
    # plt.axis([-1, 6, min(mse_train) - 0.02, max(mse_train) + 0.02])
    # plt.savefig('results/mse_lambda_train')
    # plt.show()

    # plt.plot([i for i in range(6)], mse_test, 'ko')
    # plt.axis([-1, 6, min(mse_test) - 0.02, max(mse_test) + 0.02])
    # plt.savefig('results/mse_lambda_test')
    # plt.show()


