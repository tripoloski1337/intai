from core.train import MachineLearning
from util.consolelog import Console

class Detector:
    def __init__(self, path_csv, path_model, encoded):
        self.ML = MachineLearning()
        self.path_csv = path_csv 
        self.path_model = path_model
        self.encoded = encoded
        Console.info("loading model: " + path_model)

    def ignite(self):
        self.ML.preps_predict(self.path_csv, self.path_model, self.encoded)
        Console.info("ignite model: " + self.path_model)
    
    def check(self,payload):
        return self.ML.nn_predictions(payload)