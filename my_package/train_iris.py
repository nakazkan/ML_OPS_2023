import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from my_package import load_data


def train_iris():

    X_train, X_test, y_train, y_test = load_data.load_data()

    clf = RandomForestClassifier(n_estimators=1, random_state=42)
    clf.fit(X_train, y_train)

    df = pd.DataFrame(X_test, y_test)
    df.to_csv("data/test_data.csv")

    dump(clf, "data/iris_model.joblib")
