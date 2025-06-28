class Partitioner:
    def __init__(self, splitter):
        self.splitter = splitter

    def run(self, x, y, test_size):
        x_train, x_test, y_train, y_test = self.splitter(x, y, test_size=test_size)
        return x_train, y_train, x_test, y_test
