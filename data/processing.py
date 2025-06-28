class Processor:
    def __init__(self, path, preprocessor, partitioner, resampler):
        self.path = path
        self.preprocessor = preprocessor
        self.partitioner = partitioner
        self.resampler = resampler
        self.test_size = None

    def run(self, test_size):
        x_preprocessed, y_preprocessed = self.preprocessor.run(self.path)
        x_train, y_train, x_test, y_test = self.partitioner.run(x_preprocessed, y_preprocessed, test_size)
        x_train_resampled, y_train_resampled = self.resampler.run(x_train, y_train)
        y_train_resampled = y_train_resampled.reshape(-1,1)
        return x_train_resampled, y_train_resampled, x_test, y_test