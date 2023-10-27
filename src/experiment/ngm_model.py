from abc import ABC, abstractmethod

class NGMModel(ABC):

    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def infer(self, prompt_text):
        """
        Should return a model result object that provides access to the prediction and attention.
        """
        pass


class ModelResult:

    def __init__(self, prediction, attentions):
        self.prediction = prediction
        self.attentions = attentions
