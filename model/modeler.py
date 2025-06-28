from itertools import product
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


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
        self.y_test_history = []
        self.weight_history = []
        self.biases_history = []
        self.loss_history = []
        self.prediction_history = []

    def processData(self, test_size):
        x_train, y_train, x_test, y_test =  self.data_processor.run(test_size=test_size)
        self.input_size = len(x_train[0])
        return x_train, y_train, x_test, y_test

    def run(self):
        params = self.createParamsMatrix()
        for i, param in enumerate(params):
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

            prediction = self.model.predict(x_test, self.threshold)

            self.prediction_history.append(prediction)
            self.weight_history.append(weight)
            self.biases_history.append(biases)
            self.loss_history.append(losses)
            self.y_test_history.append(y_test)
            self.model.reset()
        self.plotResults(params)

    def plotResults(self, params):
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, param in enumerate(params):
            test_size = param['test_size']
            hidden = param['hidden']
            lr = param['lr']
            epoch = param['epoch']
            batch_size = param['bs']
            losses = self.loss_history[i]
            y_predict = self.prediction_history[i]
            print(f'Y predict: {y_predict}')
            y_test = self.y_test_history[i]
            print(f'Y test: {y_test}')

            label = (
                f"Hidden: {hidden}, LR: {lr}, Epochs: {epoch}, Batch: {batch_size}, "
                f"Acc: {accuracy_score(y_test, y_predict):.3f}, "
                f"Precision: {precision_score(y_test, y_predict):.3f}, Recall: {recall_score(y_test, y_predict):.3f}"
                f"F1: {f1_score(y_test, y_predict):.3f}, ROC AUC: {roc_auc_score(y_test, y_predict):.3f}"
            )
            ax.plot(losses, label=label)
        ax.set_xlabel("Batch Updates")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves for Multiple Hyperparameters", fontsize=14)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.show()

    def createParamsMatrix(self):
        params = []
        for test_size, hidden, lr, epoch, bs in product(self.test_size, self.hidden_size, self.learning_rate, self.epochs, self.batch_size):
            param = {
                'test_size': test_size,
                'hidden': hidden,
                'lr': lr,
                'epoch': epoch,
                'bs': bs
            }
            params.append(param)
        return params

