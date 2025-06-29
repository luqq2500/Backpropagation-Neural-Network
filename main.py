from data.dependencies import dataProcessorDependencies
from model.modeler import Modeler
from model.loss_function import binaryCrossEntropyLoss
from model.activation import sigmoid, sigmoid_derivative
from model.bpnn import BackPropagationNeuralNetwork

# Hyperparameter - Variations Format: [variation, variation, ...]
TEST_SIZE = [0.3]              # Test size of data.
HIDDEN_SIZE = [[8]]           # Number of hidden layer neurons.
LEARNING_RATE = [0.1, 0.001]         # List of learning rate to adjust weight updates.
BATCH_SIZE = [2000]           # List of number of rows trained per batch
EPOCHS = [50]                  # List of epochs models update the weights.

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
parameters, losses_history, accuracy_history = modeler.run()

# Display parameter and its loss
print(f'\n======== HYPERPARAMETER TUNING RESULT ========\n')
for i, (param, losses, accuracy) in enumerate(zip(parameters, losses_history, accuracy_history)):
    print(f'Experiment: {i+1}')
    print(f'Parameter: {param}')
    print(f'Losses: {losses}')
    print(f'Accuracy: {accuracy}\n')