import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from iris import iris_classifier


def main():
    clf = iris_classifier.IrisClassifier()
    clf = clf.load_model("data/iris_model.pkl")
    df = pd.read_csv("data/test.csv")
    X_test = np.array(df)
    y_test = X_test[:, 0]
    X_test = X_test[:, 1:]

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    out = pd.DataFrame(y_pred)
    out.to_csv("data/out.csv")


if __name__ == "__main__":
    main()
