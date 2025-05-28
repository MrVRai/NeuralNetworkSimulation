# main.py

from data.load_data import load_mnist
from models.dnn import DeepNeuralNetwork
import json

def main():
    print("Starting training...")
    x_train, y_train, x_test, y_test = load_mnist(pca_components=30)
    print("Data loaded")

    dnn = DeepNeuralNetwork(sizes=[30, 32, 10], activation='relu')
    print("Model initialized")

    dnn.train(
        x_train, y_train,
        x_test, y_test,
        epochs=10,
        batch_size=64,
        optimizer='momentum',
        l_rate=0.1,
        beta=0.9
    )
    print("Training finished")
    # dnn.animate_weight_evolution('W1', save_path='W1_evolution.gif')

    dnn.export_weight_history_to_json("weights.json")

    # dnn.visualize_network("visualization/network_visualization.html")
    print("Visualization done")


if __name__ == "__main__":
    main()
