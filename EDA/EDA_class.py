import pandas as pd
import numpy as np



class EDA_class:
    def __init__(self):
        """
        Initialize the EDA class.
        """
        self.dataset = None

    def get_data_info(self, file) -> None:
        """
        Get the information of the dataset.
        """
        self.dataset = file
        print("Dataset Information:")
        print(f"Number of rows: {self.dataset.data.shape[0]}")
        print(f"Number of columns: {self.dataset.data.shape[1]}")
        print(f"Columns: {self.dataset.data.columns.tolist()}")
        print(f"Data types:\n{self.dataset.data.dtypes}")
        print(f"Missing values:\n{self.dataset.data.isnull().sum()}")

    