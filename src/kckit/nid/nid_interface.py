from abc import ABC, abstractmethod

class InformationDistanceCalculator(ABC):
    """
    Abstract class (an interface) for information distance calculation.
    """

    @abstractmethod
    def compute_distance(self, data1, data2):
        """
        Compute the distance between two data strings.
        """
        pass

    @abstractmethod
    def train_dictionary(self, samples):
        """
        Optional method for algorithms that require dictionary training.
        """
        pass