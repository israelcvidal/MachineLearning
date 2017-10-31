import numpy as np
from sklearn.utils import shuffle


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
                error = train_[-1] - outputs

                epoch_error.append(error ** 2)

                # Now we are starting back propogation!

                # Transpose hidden <-> output weights
                who_t = self.who.transpose()
                # Hidden errors is output error multiplied by weights (who)
                hidden_errors = np.dot(who_t, error)

                # Calculate the gradient, this is much nicer in python!
                gradient_output = d_sigmoid(outputs)*error









                # // Gradients for next layer, more back propogation!
                var gradient_hidden = Matrix.map(hidden_outputs, this.derivative);
                # // Weight by errors and learning rate
                gradient_hidden.multiply(hidden_errors);
                gradient_hidden.multiply(this.lr);

                # // Change in weights from HIDDEN --> OUTPUT
                var hidden_outputs_T = hidden_outputs.transpose();
                var deltaW_output = Matrix.dot(gradient_output, hidden_outputs_T);
                this.who.add(deltaW_output);

                # // Change in weights from INPUT --> HIDDEN
                var inputs_T = inputs.transpose();
                var deltaW_hidden = Matrix.dot(gradient_hidden, inputs_T);
                this.wih.add(deltaW_hidden);


                w_[0] += (alpha_ * error)
                for i in range(len(row) - 1):
                    w_[i + 1] += (alpha_ * error * row[i])

            error_history.append(sum(epoch_error) / len(epoch_error))
        return w_, error_history



