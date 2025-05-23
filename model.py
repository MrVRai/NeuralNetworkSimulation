import numpy as np
import time

class DeepNeuralNetwork():
    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'sigmoid'.")
        self.params = self.initialize()
        self.cache = {}

    def relu(self, x, derivative=False):
        if derivative:
            return np.where(x >= 0, 1, 0)
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        if derivative:
            return sig * (1 - sig)
        return sig

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def initialize(self):
        input_layer, hidden_layer, output_layer = self.sizes
        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1. / input_layer),
            "b1": np.zeros((hidden_layer, 1)),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1. / hidden_layer),
            "b2": np.zeros((output_layer, 1))
        }
        return params

    def initialize_momentum_optimizer(self):
        return {key: np.zeros_like(value) for key, value in self.params.items()}

    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], x.T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        return self.cache["A2"]

    def back_propagate(self, y, output):
        m = y.shape[0]
        dZ2 = output - y.T
        dW2 = (1. / m) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dW1 = (1. / m) * np.matmul(dZ1, self.cache["X"])
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads

    def cross_entropy_loss(self, y, output):
        m = y.shape[0]
        log_likelihood = np.multiply(y.T, np.log(output + 1e-8))  # Added epsilon to prevent log(0)
        loss = -(1. / m) * np.sum(log_likelihood)
        return loss

    def optimize(self, l_rate=0.1, beta=0.9):
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] -= l_rate * self.grads[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momentum_opt[key] = beta * self.momentum_opt[key] + (1 - beta) * self.grads[key]
                self.params[key] -= l_rate * self.momentum_opt[key]
        else:
            raise ValueError("Unsupported optimizer. Use 'sgd' or 'momentum'.")

    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))

    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, optimizer='momentum', l_rate=0.1, beta=0.9):
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // batch_size)
        self.optimizer = optimizer
        if optimizer == 'momentum':
            self.momentum_opt = self.initialize_momentum_optimizer()

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"

        for i in range(epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                begin = j * batch_size
                end = min(begin + batch_size, x_train.shape[0])
                x_batch = x_train_shuffled[begin:end]
                y_batch = y_train_shuffled[begin:end]

                output = self.feed_forward(x_batch)
                self.back_propagate(y_batch, output)
                self.optimize(l_rate=l_rate, beta=beta)

            output_train = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output_train)
            train_loss = self.cross_entropy_loss(y_train, output_train)

            output_test = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output_test)
            test_loss = self.cross_entropy_loss(y_test, output_test)

            print(template.format(i + 1, time.time() - start_time, train_acc, train_loss, test_acc, test_loss))
