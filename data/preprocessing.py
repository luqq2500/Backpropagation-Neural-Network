import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self):
        self.data = None

    def run(self, filepath):
        self.data = pd.read_csv(filepath)
        self.cleaning()
        y_preprocessed = self.setTarget()
        x_preprocessed = self.normalize()
        return x_preprocessed, y_preprocessed

    def cleaning(self):
        self.data = (
            self.data
            .drop(columns=['ID'])
            .dropna()
            .drop_duplicates()
        )
        return self

    def normalize(self):
        x_numeric = self.data.select_dtypes(include=np.number).values
        x_preprocessed = (x_numeric - x_numeric.mean(axis=0)) / x_numeric.std(axis=0)
        return x_preprocessed

    def setTarget(self):
        y_preprocessed = self.data['Target'].values.reshape(-1, 1)
        return y_preprocessed

    # def saveToFile(self, filename: str = 'preprocessed_data.csv'):
    #     output_dir = 'data/preprocessed'
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     name, ext = os.path.splitext(filename)
    #     version = 1
    #     versioned_filename = f"{name}_v{version}{ext}"
    #     full_path = os.path.join(output_dir, versioned_filename)
    #
    #     # Find next available version number
    #     while os.path.exists(full_path):
    #         version += 1
    #         versioned_filename = f"{name}_v{version}{ext}"
    #         full_path = os.path.join(output_dir, versioned_filename)
    #
    #     self.data.to_csv(full_path, index=False)