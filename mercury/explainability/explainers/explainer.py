from abc import ABC, abstractmethod

import os
import dill

class MercuryExplainer(ABC):

    @abstractmethod
    def explain(self, data):
        pass

    def save(self, filename: str = "explainer.pkl"):
        """
        Saves the explainer with its internal state to a file.

        Args:
            filename (str): Path where the explainer will be saved
        """
        with open(filename, 'wb') as f:
            dill.dump(self, f)
        assert os.path.isfile(filename), "Error storing file"

    @classmethod
    def load(self, filename: str = "explainer.pkl"):
        """
        Loads a previosly saved explainer with its internal state to a file.

        Args:
            filename (str): Path where the explainer is stored
        """
        assert os.path.isfile(filename), "File does not exist or not a valid file"
        with open(filename, 'rb') as f:
            return dill.load(f)

