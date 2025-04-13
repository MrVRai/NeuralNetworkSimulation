import numpy as np

class DeepNeuralNetwork():
    def __init__(self, sizes):
        self.sizes = sizes
        self.params = self.initialize()
        self.cache = {}

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
        return {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1. / input_layer),
            "b1": np.zeros((hidden_layer, 1)),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1. / hidden_layer),
            "b2": np.zeros((output_layer, 1))
        }

    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], x.T) + self.params["b1"]
        self.cache["A1"] = self.sigmoid(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        return self.cache["A2"]

    def cross_entropy_loss(self, y, output):
        m = y.shape[0]
        log_probs = np.multiply(y.T, np.log(output + 1e-9))  # Add small epsilon for numerical stability
        loss = -np.sum(log_probs) / m
        return loss

    def accuracy(self, y, output):
        predictions = np.argmax(output, axis=0)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)

    def back_propagate(self, y, output):
        m = y.shape[0]
        dZ2 = output - y.T
        dW2 = (1. / m) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.sigmoid(self.cache["Z1"], derivative=True)
        dW1 = (1. / m) * np.matmul(dZ1, self.cache["X"])
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads
