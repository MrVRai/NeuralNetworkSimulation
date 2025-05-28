#models/dnn.py

import numpy as np
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

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
        log_likelihood = np.multiply(y.T, np.log(output + 1e-8))  # epsilon prevents log(0)
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

    def train(self, x_train, y_train, x_test, y_test,
              epochs=10, batch_size=64,
              optimizer='momentum', l_rate=0.1, beta=0.9,
              visualize=False, save_html_path=None):
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // batch_size)
        self.optimizer = optimizer
        if optimizer == 'momentum':
            self.momentum_opt = self.initialize_momentum_optimizer()

        template = "Epoch {}: train acc={:.4f}, train loss={:.4f}, test acc={:.4f}, test loss={:.4f}"

        self.weight_history = {"W1": [], "W2": []}
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

            self.weight_history["W1"].append(self.params["W1"].copy())
            self.weight_history["W2"].append(self.params["W2"].copy())

            print(template.format(i + 1, train_acc, train_loss, test_acc, test_loss))

            if visualize and save_html_path is not None:
                self.visualize_network(save_html_path + f"_epoch_{i+1}.html")

    def visualize_network(self, save_html_path="visualizations/network_visualization.html"):
        sizes = self.sizes
        W1 = self.params['W1']
        W2 = self.params['W2']

        # Sample input neurons to avoid clutter
        input_sample_step = 20
        input_indices = np.arange(0, sizes[0], input_sample_step)
        x_input = np.zeros(len(input_indices))
        y_input = np.linspace(0, 1, len(input_indices))

        # Use all neurons for hidden and output layers
        x_hidden = np.ones(sizes[1]) * 1.5  # more horizontal spacing
        y_hidden = np.linspace(0, 1, sizes[1])

        x_output = np.ones(sizes[2]) * 3
        y_output = np.linspace(0, 1, sizes[2])

        layers = [
            (x_input, y_input),
            (x_hidden, y_hidden),
            (x_output, y_output)
        ]

        fig = go.Figure()

        # Draw neurons
        for layer_idx, (x_coords, y_coords) in enumerate(layers):
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(size=20, color='lightblue', line=dict(color='black', width=1.5)),
                name=f'Layer {layer_idx + 1} Neurons',
                hoverinfo='text',
                text=[f'Layer {layer_idx + 1} Neuron {i+1}' for i in range(len(x_coords))]
            ))

        # Draw connections W1 (input to hidden) - only sampled inputs & significant weights
        for i in range(W1.shape[0]):       # hidden neurons
            for idx, j in enumerate(input_indices):   # sampled input neurons
                weight = W1[i, j]
                if abs(weight) < 0.05:  # skip small weights
                    continue
                color = 'blue' if weight > 0 else 'red'
                opacity = min(1, abs(weight)*5)
                fig.add_trace(go.Scatter(
                    x=[layers[0][0][idx], layers[1][0][i]],
                    y=[layers[0][1][idx], layers[1][1][i]],
                    mode='lines',
                    line=dict(color=color, width=1),
                    opacity=opacity,
                    hoverinfo='text',
                    text=[f'W1[{i},{j}] = {weight:.4f}']
                ))

        # Draw connections W2 (hidden to output) - all neurons, significant weights only
        for i in range(W2.shape[0]):       # output neurons
            for j in range(W2.shape[1]):   # hidden neurons
                weight = W2[i, j]
                if abs(weight) < 0.05:
                    continue
                color = 'blue' if weight > 0 else 'red'
                opacity = min(1, abs(weight)*5)
                fig.add_trace(go.Scatter(
                    x=[layers[1][0][j], layers[2][0][i]],
                    y=[layers[1][1][j], layers[2][1][i]],
                    mode='lines',
                    line=dict(color=color, width=1),
                    opacity=opacity,
                    hoverinfo='text',
                    text=[f'W2[{i},{j}] = {weight:.4f}']
                ))

        fig.update_layout(
            title='Neural Network Visualization',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            height=600,
            width=900,
            plot_bgcolor='white'
        )

        # Create output folder if it doesn't exist
        folder = os.path.dirname(save_html_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        fig.write_html(save_html_path)
        print(f"Saved network visualization to {save_html_path}")

    def animate_weight_evolution(self, matrix='W1', save_path=None):

        if matrix not in self.weight_history:
            print(f"No weight history found for {matrix}")
            return

        fig, ax = plt.subplots()
        weight_frames = self.weight_history[matrix]
        vmin = -np.max(np.abs(weight_frames))  # symmetric color scale
        vmax = np.max(np.abs(weight_frames))

        def update(i):
            ax.clear()
            im = ax.imshow(weight_frames[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax.set_title(f'{matrix} Weights at Epoch {i + 1}')
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(weight_frames), blit=True, repeat=False)

        if save_path:
            ani.save(save_path, writer='pillow')
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

    def export_weight_history_to_json(self, save_path="weights.json"):
        data = {
                "W1": [w.T.tolist() for w in self.weight_history["W1"]],  # transpose for (input, hidden)
                "W2": [w.T.tolist() for w in self.weight_history["W2"]]   # transpose for (hidden, output)
            
        }
        with open(save_path, 'w') as f:
            json.dump(data, f)
        print(f"Weight history exported to {save_path}")


            