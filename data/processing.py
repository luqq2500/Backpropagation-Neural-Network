from data.partitioning import partition
from data.preprocessing import DataPreprocessing

def processing(data_path, test_size):
    preprocessor = DataPreprocessing(data_path)
    x_preprocessed, y_preprocessed = preprocessor.execute()
    x_train, y_train, x_test, y_test = partition(x_preprocessed, y_preprocessed,test_size)
    return x_train, y_train, x_test, y_test