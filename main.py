from data.dependencies import dataProcessorDependencies
from model.modeler import Modeler
from model.loss_function import binaryCrossEntropyLoss
from model.activation import sigmoid, sigmoid_derivative
from model.bpnn import BackPropagationNeuralNetwork

# Hyperparameter Variations
TEST_SIZE = [0.4, 0.3]               # Test size of data.
HIDDEN_SIZE = [[8],[4,8]]     # List of number of hidden layer neurons.
LEARNING_RATE = [0.01, 0.1]           # List of learning rate to adjust weight updates.
BATCH_SIZE = [2000, 4000]              # List of number of rows trained per batch
EPOCHS = [50, 100]                   # List of epochs models update the weights.

processor = dataProcessorDependencies()
model = BackPropagationNeuralNetwork()
activation = sigmoid
activation_deriv = sigmoid_derivative
loss = binaryCrossEntropyLoss
modeler = Modeler(
    data_processor=processor,
    test_size=TEST_SIZE,
    model=model,
    activation=activation,
    activation_deriv=activation_deriv,
    loss=loss,
    hidden_size=HIDDEN_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE)

# Run modeler
modeler.run()