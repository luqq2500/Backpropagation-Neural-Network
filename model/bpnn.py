import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from model.activation import sigmoid, sigmoid_derivative
from model.loss_function import binaryCrossEntropyLoss


class BackPropagationNeuralNetwork:
    def __init__(self):
        self.activation = None
        self.loss = None
        self.learning_rate = None
        self.epochs = None
        self.batch_size = None
        self.b2 = None
        self.W2 = None
        self.b1 = None
        self.W1 = None
        self.a2 = None
        self.z2 = None
        self.a1 = None
        self.z1 = None
        self.losses = []
        self.accuracy = 0.00

    def fit(self, x, y):
        for epoch in range(self.epochs):
            print(x.shape, y.shape)
            indices = np.random.permutation(x.shape[0])
            X_shuffled = x[indices]
            y_shuffled = y[indices]
            for i in range(0, x.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                y_pred = self.feedForward(X_batch)
                loss = binaryCrossEntropyLoss(y_batch, y_pred)
                self.losses.append(loss)
                self.backPropagate(X_batch, y_batch, y_pred)
            # if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}")
        return self

    def feedForward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backPropagate(self, X, y_true, y_pred):
        m = X.shape[0]
        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def predict(self, x, threshold):
        a1 = self.activation(np.dot(x, self.W1) + self.b1)
        a2 = self.activation(np.dot(a1, self.W2) + self.b2)
        return (a2>threshold).astype(int)

    def test(self, x_test, y_test, threshold):
        result = self.predict(x_test, threshold)
        self.accuracy = np.mean(result == y_test)
        print("Accuracy:", accuracy_score(y_test, result))
        print("Precision:", precision_score(y_test, result))
        print("Recall:", recall_score(y_test, result))
        print("F1 Score:", f1_score(y_test, result))
        print("Confusion Matrix:\n", confusion_matrix(y_test, result))
        print("Classification Report:\n", classification_report(y_test, result))
        return self

    def build(self):
        return self.W1, self.b1, self.W2, self.b2, self.losses, self.accuracy

    def configureFunction(self, activation, loss):
        self.activation = activation
        self.loss = loss
        return self

    def configureTrainingParameters(self, learning_rate, epochs, batch_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        return self

    def configureNeuron(self, input_size, hidden_size, output_size, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        return self


