from data.partitioning import partition
from data.preprocessing import DataPreprocessing

file_path = 'data/source/credit_card_eligibility.csv'

preprocessor = DataPreprocessing(file_path)
x_preprocessed, y_preprocessed = preprocessor.execute()
TEST_SIZE = 0.4
x_train, y_train, x_test, y_test = partition(x_preprocessed, y_preprocessed,TEST_SIZE)