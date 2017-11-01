import numpy as np
from sklearn.utils import shuffle
from scipy.io import loadmat

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def d_sigmoid(x):
    return x * (1 - x)
#
# def predict(x, wih, who):
#     # The input to the hidden layer is the weights (wih) multiplied by inputs
#     hidden_inputs = np.dot(wih, x)
#     # The outputs of the hidden layer pass through sigmoid activation function
#     hidden_outputs = sigmoid(hidden_inputs)
#     # The input to the output layer is the weights (who) multiplied by hidden layer
#     output_inputs = np.dot(who, hidden_outputs)
#     outputs = sigmoid(output_inputs)
#     return outputs


class NeuralNetwork(object):
    def __init__(self, n_input, n_hidden, n_output, n_epochs, learning_rate):
        self.input_nodes = n_input
        self.hidden_nodes = n_hidden
        self.output_nodes = n_output
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # wij = peso do neuronio i da camada j
        self.wih = np.random.rand(n_hidden, n_input)
        self.who = np.random.rand(n_output, n_hidden)

    def fit(self, train_):
        train_ = np.concatenate(([[-1] for _ in range(len(train_))], train_))
        error_history = []

        for epoch in range(self.n_epochs):
            train_ = shuffle(train_)

            epoch_error = []
            for row in train_:
                # The input to the hidden layer is the weights (wih) multiplied by inputs
                hidden_inputs = np.dot(self.wih, row[:-1])
                # The outputs of the hidden layer pass through sigmoid activation function
                hidden_outputs = sigmoid(hidden_inputs)
                # The input to the output layer is the weights (who) multiplied by hidden layer
                output_inputs = np.dot(self.who, hidden_outputs)
                #  The output of the network passes through sigmoid activation function
                outputs = sigmoid(output_inputs)

                # Error is TARGET - OUTPUT
                error = row[-1] - outputs

                epoch_error.append(error ** 2)

                # Now we are starting back propogation!

                # Transpose hidden <-> output weights
                who_t = self.who.transpose()
                # Hidden errors is output error multiplied by weights (who)
                hidden_errors = np.dot(who_t, error)

                # Calculate the gradient, this is much nicer in python!
                gradient_output = d_sigmoid(outputs)*error

                # Gradients for next layer, more back propogation!
                # Weight by errors and learning rate
                gradient_hidden = d_sigmoid(hidden_outputs)*hidden_errors*self.learning_rate

                # // Change in weights from HIDDEN --> OUTPUT
                hidden_outputs_t = hidden_outputs.transpose()
                delta_w_output = np.dot(gradient_output, hidden_outputs_t)
                self.who = self.who + delta_w_output

                # // Change in weights from INPUT --> HIDDEN
                inputs_t = row.transpose()
                delta_w_hidden = np.dot(gradient_hidden, inputs_t)
                self.wih = self.wih + delta_w_hidden

            error_history.append(sum(epoch_error) / len(epoch_error))
        return self.wih, self.who, error_history


def main():
    data = loadmat("data/DadosExemplo.mat")
    x = data.get('x')
    y = data.get('y')

    print(x.shape, y.shape)


if __name__ == '__main__':
    main()


