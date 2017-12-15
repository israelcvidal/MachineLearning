import numpy as np
from sklearn.utils import shuffle
from scipy.io import loadmat


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def d_sigmoid(x):
    return x * (1 - x)


class NeuralNetwork(object):
    def __init__(self, n_input, n_hidden, n_output, n_epochs, learning_rate):
        self.input_nodes = n_input
        self.hidden_nodes = n_hidden
        self.output_nodes = n_output
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.wih = np.random.rand(n_hidden, n_input)
        self.who = np.random.rand(n_output, n_hidden)

    def fit(self, x_train, y_train):
        # x_train = np.concatenate(([[-1] for _ in range(len(x_train))], x_train), axis=1)
        error_history = []

        for epoch in range(self.n_epochs):

            epoch_error = []

            # The input to the hidden layer is the weights (wih) multiplied by inputs
            hidden_inputs = np.dot(self.wih, x_train)
            # The outputs of the hidden layer pass through sigmoid activation function
            hidden_outputs = sigmoid(hidden_inputs)
            # The input to the output layer is the weights (who) multiplied by hidden layer
            output_inputs = np.dot(self.who, hidden_outputs)
            #  The output of the network passes through sigmoid activation function
            outputs = sigmoid(output_inputs)

            # Error is TARGET - OUTPUT
            errors = y_train - outputs

            epoch_error.append(sum(errors ** 2)/len(errors))
            print(errors**2)
            # print(sum(errors**2)/len(errors))
            # Now we are starting back propagation!

            # Transpose hidden <-> output weights
            who_t = self.who.transpose()
            # Hidden errors is output errors multiplied by weights (who)
            hidden_errors = np.dot(who_t, errors)

            # Calculate the gradient
            gradient_output = d_sigmoid(outputs)*errors

            # Gradients for next layer, more back propagation!
            # Weight by errors and learning rate
            gradient_hidden = d_sigmoid(hidden_outputs)*hidden_errors

            # // Change in weights from HIDDEN --> OUTPUT
            hidden_outputs_t = hidden_outputs.transpose()
            delta_w_output = np.dot(gradient_output, hidden_outputs_t)
            self.who = self.who + (self.learning_rate*delta_w_output)

            # // Change in weights from INPUT --> HIDDEN
            inputs_t = x_train.transpose()
            delta_w_hidden = np.dot(gradient_hidden, inputs_t)
            self.wih = self.wih + (self.learning_rate*delta_w_hidden)

            error_history.append(sum(epoch_error) / len(epoch_error))
        # return self.wih, self.who, error_history
        print(self.wih.shape)
        return error_history

def main():
    data = loadmat("data/ex3data1.mat")
    x = data.get('X').transpose()
    y = data.get('T').transpose()

    # print(x.shape)
    # print(y.shape)
    nn = NeuralNetwork(400, 10, 10, 300, 0.01)
    nn.fit(x, y)

    data = loadmat("data/DadosExemplo.mat")
    x = data.get("x").transpose()
    y = data.get("y")

    # nn = NeuralNetwork(2, 2, 2, 300, 0.01)
    # nn.fit(x, y)
    # print(nn.fit(x, y))
    # print(x.shape, y.shape)


if __name__ == '__main__':
    main()


