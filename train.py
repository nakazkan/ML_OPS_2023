from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from iris import iris_classifier


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=42
    )
    df_test = pd.DataFrame(X_test, y_test)
    df_train = pd.DataFrame(X_train, y_train)

    clf = iris_classifier.IrisClassifier()
    clf.fit(X_train, y_train)

    output_file = Path("data/iris_model.pkl")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    clf.save_model("data/iris_model.pkl")

    df_test.to_csv("data/test.csv")
    df_train.to_csv("data/train.csv")


if __name__ == "__main__":
    main()
