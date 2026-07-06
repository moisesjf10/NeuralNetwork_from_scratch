import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from network import NeuralNetwork
from layers import Layer, FlattenLayer, PoolingLayer, ConvLayer
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
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test  = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test

def evaluate_accuracy(net, X, y_true_onehot, batch_size=64):
    n_samples = X.shape[0]
    correct_predictions = 0

    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y_true_onehot[i:i + batch_size]
        
        y_pred_probs = net.forward(X_batch)
        
        predictions = np.argmax(y_pred_probs, axis=1)
        trues = np.argmax(y_batch, axis=1)
        
        correct_predictions += np.sum(predictions == trues)

    return correct_predictions / n_samples

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
        ConvLayer(1, 8, 3, padding=1),
        PoolingLayer(2),           
        ConvLayer(8, 16, 3, padding=1),
        PoolingLayer(2),            
        FlattenLayer(),                  
        Layer(784, 64, ReLU()),      
        Layer(64, 10, Softmax()),   
    ],
    loss=CategoricalCrossEntropy()
)

    #Training
    EPOCHS=5
    BATCH_SIZE=64
    LEARNING_RATE=0.05

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

    net.save_weights("cnn_mnist.pkl")
    