import matplotlib.pyplot as plt
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

    for epoch in range(epochs):
        train = shuffle(train)

        epoch_error = []
        for row in train:
            error = row[-1] - predict(row[:-1], w_)
            epoch_error.append(error**2)
            w_[0] += (alpha*error)
            for i in range(len(row) - 1):
                w_[i + 1] += (alpha * error * row[i])

        error_history.append(sum(epoch_error)/(2*len(epoch_error)))
    return w_, error_history

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
    alpha = 0.01
    epochs = 100

    data2 = np.loadtxt("data/ex1data2.txt")


    # line = [predict([i], [-3.917517058471562, 1.303534229171897]) for i in x]

    w, epoch_errors = sgd(train=data2, alpha=alpha, epochs=epochs)

    # show results for dataset2:
    print("w: " + str(w))
    plt.plot([i for i in range(0, epochs)], epoch_errors, 'k-')
    # plt.show()

    # save results for dataset2
    plt.savefig('results/epochs_mse_dataset2')
