from model.develop import ModelDeveloper
from model.loss_function import binaryCrossEntropyLoss
from model.activation import sigmoid
from model.bpnn import BackPropagationNeuralNetwork
from data.processing import x_train, y_train, x_test, y_test

INPUT_SIZE  = x_train.shape[1]
OUTPUT_SIZE = 1

# HYPERPARAMETER CONFIGURATION
HIDDEN_SIZE = [16, 32, 64] # Number of hidden neurons.
LEARNING_RATE = [0.1, 0.3, 0.5] # Learning rate to adjust weight updates.
BATCH_SIZE = [1500] # Number of rows trained per batch
EPOCHS = [50] # Epochs models update the weights.
THRESHOLD = [0.7] # Prediction threshold.

model = BackPropagationNeuralNetwork()
activation = sigmoid
loss = binaryCrossEntropyLoss

developer = ModelDeveloper(
    model=model,
    activation=activation,
    loss=loss,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    threshold=THRESHOLD)

developer.run(x_train, y_train, x_test, y_test)