import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class IrisClassifier:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=1, random_state=42)

    def fit(self, data: np.ndarray, target: np.ndarray):
        self.clf.fit(data, target)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.clf.predict(data)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)
        return self.clf

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)
