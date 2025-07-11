from itertools import product
from matplotlib import pyplot as plt
from model.performance import measurePerformanceMetrics

# Modeler class responsible for modeling variations of parameter, for experimentation.
class Modeler:
    def __init__(self, data_processor, test_size, model, activation, activation_deriv, loss, hidden_size, learning_rate, epochs, batch_size):
        self.data_processor = data_processor
        self.test_size = test_size
        self.model = model
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.loss = loss
        self.input_size = None
        self.hidden_size = hidden_size
        self.output_size = 1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = 0.5
        self.params = []
        self.weight_history = []
        self.biases_history = []
        self.loss_history = []
        self.y_test_history = []
        self.y_predict_history = []
        self.accuracy_history = []

    # Loop through each parameter combination to model, test, and measure loss.
    def run(self):
        self.createParameterCombinationMatrix()
        for i, param in enumerate(self.params):
            test_size = param['test_size']
            hidden_size = param['hidden']
            print(hidden_size)
            lr = param['lr']
            epoch = param['epoch']
            batch_size = param['bs']

            x_train, y_train, x_test, y_test = self.processData(test_size)

            weight, biases, losses = ((((self.model
                .configureNetwork(input_size=self.input_size, hidden_size=hidden_size, output_size=self.output_size)
                .configureTraining(learning_rate=lr, epochs=epoch, batch_size=batch_size))
                .configureFunctions(activation=self.activation, activation_deriv=self.activation_deriv,loss=self.loss))
                .fit(x_train, y_train)
                .getParameters()))

            y_predict = self.model.predict(x_test, self.threshold)
            accuracy = measurePerformanceMetrics(y_test, y_predict)

            self.weight_history.append(weight)
            self.biases_history.append(biases)
            self.loss_history.append(losses)
            self.y_predict_history.append(y_predict)
            self.y_test_history.append(y_test)
            self.accuracy_history.append(round(accuracy, 3))
            self.model.reset()
        self.plotResults()
        return self.params, self.loss_history, self.accuracy_history

    # Process data with test size variations.
    def processData(self, test_size):
        x_train, y_train, x_test, y_test =  self.data_processor.run(test_size=test_size)
        self.input_size = len(x_train[0])
        return x_train, y_train, x_test, y_test

    # Create iterable parameter combinations.
    def createParameterCombinationMatrix(self):
        for test_size, hidden, lr, epoch, bs in product(self.test_size, self.hidden_size, self.learning_rate, self.epochs, self.batch_size):
            parameter = {
                'test_size': test_size,
                'hidden': hidden,
                'lr': lr,
                'epoch': epoch,
                'bs': bs
            }
            self.params.append(parameter)

    # Plot each model variation's parameters and performances.
    def plotResults(self):
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, param in enumerate(self.params):
            losses = self.loss_history[i]
            accuracy = self.accuracy_history[i]
            # y_predict = self.y_predict_history[i]
            # y_test = self.y_test_history[i]
            # accuracy = measurePerformanceMetrics(y_test, y_predict)
            label = (
                f"Hidden: {param['hidden']}, LR: {param['lr']}, "
                f"Epochs: {param['epoch']}, Batch: {param['bs']}, "
                f"Test Size: {param['test_size']}, Acc: {accuracy:.3f}, "
                # f"Precision: {precision:.3f}, Recall: {recall:.3f}"
                # f"F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}"
            )
            ax.plot(losses, label=label)
        ax.set_xlabel("Batch Updates")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves for Multiple Hyperparameters", fontsize=14)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
        plt.tight_layout()
        plt.show()


