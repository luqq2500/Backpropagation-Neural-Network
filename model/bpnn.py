import numpy as np

class BackPropagationNeuralNetwork:
    def __init__(self):
        self.activation = None
        self.activation_deriv = None
        self.loss = None
        self.learning_rate = None
        self.epochs = None
        self.batch_size = None
        self.weights = []
        self.biases = []
        self.zs = []
        self.activations = []
        self.losses = []
        self.accuracy = 0.0

    def fit(self, X, y):
        loss = 0
        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            # shuffle
            indices = np.random.permutation(n_samples)
            X_sh, y_sh = X[indices], y[indices]
            for start in range(0, n_samples, self.batch_size):
                X_batch = X_sh[start:start + self.batch_size]
                y_batch = y_sh[start:start + self.batch_size]
                y_pred = self.feedForward(X_batch)
                loss = self.loss(y_batch, y_pred)
                self.losses.append(loss)
                self.backPropagate(y_batch, y_pred)
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}")
        return self

    def feedForward(self, X):
        """Perform forward pass through all layers"""
        self.zs = []
        self.activations = [X]
        a = X
        for W, b in zip(self.weights, self.biases):
            z = np.dot(a, W) + b
            a = self.activation(z)
            self.zs.append(z)
            self.activations.append(a)
        return a

    def backPropagate(self, y_true, y_pred):
        """Perform backpropagation and update weights and biases"""
        m = y_true.shape[0]
        # initialize gradient lists
        grad_w = [np.zeros_like(W) for W in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # delta for output layer
        delta = (y_pred - y_true)
        print(f'Delta: {delta}')# for BCE with sigmoid
        # backprop through layers
        for l in reversed(range(len(self.weights))):
            a_prev = self.activations[l]
            grad_w[l] = np.dot(a_prev.T, delta) / m
            grad_b[l] = np.sum(delta, axis=0, keepdims=True) / m
            # compute delta for next (previous) layer
            if l > 0:
                z_prev = self.zs[l-1]
                delta = np.dot(delta, self.weights[l].T) * self.activation_deriv(z_prev)

        # update parameters
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

    def predict(self, X, threshold:float):
        a = X
        for W, b in zip(self.weights, self.biases):
            a = self.activation(np.dot(a, W) + b)
        return (a>threshold).astype(int)

    def configureFunctions(self, activation, activation_deriv, loss):
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.loss = loss
        return self

    def configureTraining(self, learning_rate, epochs, batch_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        return self

    def configureNetwork(self, input_size, hidden_size, output_size, seed=42):
        """
        hidden_size: list of neuron counts for each hidden layer, e.g. [n1, n2, ...]
        """
        np.random.seed(seed)
        # build layer sizes sequence: input -> hidden_size... -> output
        layer_sizes = [input_size] + list(hidden_size) + [output_size]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        return self

    def reset(self):
        self.__init__()

    def getParameters(self):
        return self.weights, self.biases, self.losses
