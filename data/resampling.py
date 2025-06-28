class Resampler:
    def __init__(self, resampler):
        self.resampler = resampler

    def run(self, x,y):
        x_resampled, y_resampled = self.resampler.fit_resample(x, y)
        return x_resampled, y_resampled