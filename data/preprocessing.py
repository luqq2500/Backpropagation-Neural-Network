import numpy as np
import pandas as pd
import os

class DataPreprocessing:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        self.x_preprocessed = None
        self.y_preprocessed = None

    def execute(self):
        self.cleaning()
        self.setTarget()
        self.normalize()
        self.saveToFile()
        return self.x_preprocessed, self.y_preprocessed

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
        self.x_preprocessed = (x_numeric - x_numeric.mean(axis=0)) / x_numeric.std(axis=0)

    def setTarget(self):
        self.y_preprocessed = self.data['Target'].values.reshape(-1, 1)

    def saveToFile(self, filename: str = 'preprocessed_data.csv'):
        output_dir = 'data/preprocessed'
        os.makedirs(output_dir, exist_ok=True)

        name, ext = os.path.splitext(filename)
        version = 1
        versioned_filename = f"{name}_v{version}{ext}"
        full_path = os.path.join(output_dir, versioned_filename)

        # Find next available version number
        while os.path.exists(full_path):
            version += 1
            versioned_filename = f"{name}_v{version}{ext}"
            full_path = os.path.join(output_dir, versioned_filename)

        self.data.to_csv(full_path, index=False)








