import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from network import NeuralNetwork
from layers import Layer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy

def load_and_preprocess_mnist():
    #Load MNIST dataset 
    print("Loading MNIST dataset...")
    mnist=fetch_openml('mnist_784', version=1, parser='auto')

    X=mnist.data.values.astype(np.float32)/255.0    
    y=mnist.target.values.astype(np.int32).reshape(-1, 1)

    print("Preprocessing data...")

    encoder=OneHotEncoder(sparse_output=False)
    y_onehot=encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test=train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def evaluate_accuracy(net, X, y_true_onehot):
    y_pred_probs=net.forward(X)

    predictions=np.argmax(y_pred_probs, axis=1)
    trues=np.argmax(y_true_onehot, axis=1)

    accuracy=np.mean(predictions==trues)

    return accuracy

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_mnist()

    print(f"\nData ready")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    #Net architecture
    # Input: 784 neurons (28x28 pixels)
    # Hidden layer 1: 128 neurons (ReLU)
    # Hidden layer 2: 64 neurons (ReLU)
    # Output layer : 10 neurons (Softmax)

    print(f"\nInitializing the net")
    net = NeuralNetwork(
        layers=[
            Layer(784, 128, ReLU()),
            Layer(128, 64, ReLU()),
            Layer(64,10, Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )

    #Training
    EPOCHS=30
    BATCH_SIZE=128
    LEARNING_RATE=0.1

    print("\nStarting the training...")

    losses=net.train(X=X_train, y=y_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    lr=LEARNING_RATE)

    #Evaluation
    print("\n Evaluating the model")
    train_acc=evaluate_accuracy(net, X_train, y_train)
    test_acc=evaluate_accuracy(net, X_test, y_test)

    print("-" * 30)
    print(f"Trainig Accuracy: {train_acc * 100:.3f}%")
    print(f"Test Accuracy: {test_acc * 100:.3f}%")
    print("-" * 30)
    