import numpy as np
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

    def build(self):
        return self.W1, self.b1, self.W2, self.b2, self.losses

    def reset(self):
        self.__init__()

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


