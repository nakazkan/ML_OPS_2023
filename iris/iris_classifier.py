import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class IrisClassifier(RandomForestClassifier):
    def __init__(
        self,
        n_estimators: int = 1,
        random_state: int = 0,
        max_depth: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, max_depth=max_depth
        )

    def fit(self, data: np.ndarray, target: np.ndarray):
        self.clf.fit(data, target)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.clf.predict(data)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(data)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)
        return self.clf

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)

    def get_classifier(self) -> RandomForestClassifier:
        return self.clf
