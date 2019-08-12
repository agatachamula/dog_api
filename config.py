

class DogConfig:
    PORT = "5005"
    MODEL_PATH = "trained_model/retrained_graph.pb"
    LABELS_PATH = "trained_model/retrained_labels.txt"
    APP_NAME = "DogApp"
    DEBUG = "True"
    HOST = "127.0.0.1"
    CACHE_TYPE = "simple"


class ListURL:
    PredictBreedURL = "/predict/breed"
    RootURL="/"

