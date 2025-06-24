from itertools import product
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class ModelDeveloper:
    def __init__(self, model, activation, loss, input_size, hidden_size, output_size, learning_rate, epochs, batch_size, threshold):
        self.model = model
        self.activation = activation
        self.loss = loss
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.weight_bias_history = []
        self.loss_history = []
        self.prediction_history = []

    def run(self, xtrain, ytrain, xtest, ytest):
        params = self.createParamsMatrix()
        for i, param in enumerate(params):
            hidden = param['hidden']
            lr = param['lr']
            epoch = param['epoch']
            batch_size = param['bs']
            threshold = param['threshold']

            w1, b1, w2, b2, losses = ((((self.model.configureNeuron(input_size=self.input_size, hidden_size=hidden, output_size=self.output_size)
                .configureTrainingParameters(learning_rate=lr, epochs=epoch, batch_size=batch_size))
                .configureFunction(activation=self.activation, loss=self.loss))
                .fit(xtrain, ytrain)
                .build()))

            prediction = self.model.predict(xtest, threshold)

            self.prediction_history.append(prediction)
            self.weight_bias_history.append([w1, b1, w2, b2])
            self.loss_history.append(losses)
            self.model.reset()
        self.plotResults(params, ytest)

    def createParamsMatrix(self):
        params = []
        for hidden, lr, epoch, bs, threshold in product(self.hidden_size, self.learning_rate, self.epochs, self.batch_size, self.threshold):
            param = {
                'hidden': hidden,
                'lr': lr,
                'epoch': epoch,
                'bs': bs,
                'threshold': threshold,
            }
            params.append(param)
        return params

    def plotResults(self, params, y_test):
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, param in enumerate(params):
            hidden = param['hidden']
            lr = param['lr']
            epoch = param['epoch']
            batch_size = param['bs']
            losses = self.loss_history[i]
            y_predict = self.prediction_history[i]
            label = (
                f"Hidden: {hidden}, LR: {lr}, Epochs: {epoch}, "
                f"Batch size: {batch_size}, Accuracy : {accuracy_score(y_test, y_predict):.3f}",
                f"F1 Score: {f1_score(y_test, y_predict):.3f}, ROC AUC: {roc_auc_score(y_test, y_predict):.3f}"
            )
            ax.plot(losses, label=label)
        ax.set_xlabel("Batch Updates")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves for Multiple Hyperparameters", fontsize=14)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.show()

