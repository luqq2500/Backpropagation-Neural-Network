from data.preprocessing import DataPreprocessing
from data.partitioning import partition
from model.bpnn import BackPropagationNeuralNetwork
from model.activation import sigmoid
from model.loss_function import binaryCrossEntropyLoss
from model.plot import plotResult

HIDDEN_SIZE = 3
OUTPUT_SIZE = 1
LEARNING_RATE = [0.01]
BATCH_SIZE = 3000
EPOCHS = 10
TEST_SIZE = 0.4
THRESHOLD = 0.7

file_path = 'data/raw\credit_card_eligibility.csv'

preprocessor = DataPreprocessing(file_path)
x_preprocessed, y_preprocessed = preprocessor.execute()

x_train, y_train, x_test, y_test = partition(x_preprocessed, y_preprocessed,TEST_SIZE)

input_size  = x_train.shape[1]


activation = sigmoid
loss = binaryCrossEntropyLoss

for lr in LEARNING_RATE:
 model = BackPropagationNeuralNetwork()
 w1, b1, w2, b2, losses, accuracy = (model.configureFunction(activation=activation, loss=loss)
  .configureNeuron(input_size=input_size, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
  .configureTrainingParameters(learning_rate=lr, epochs=EPOCHS, batch_size=BATCH_SIZE)
  .fit(x_train, y_train)
  .test(x_test, y_test, THRESHOLD)
  .build()
  )
 plotResult(losses, accuracy, activation, LEARNING_RATE, EPOCHS, BATCH_SIZE)
 print(f'Developed weight for layer 1: {w1}, weight for layer 2: {w2}, bias 1: {b1}, bias 2: {b2}')