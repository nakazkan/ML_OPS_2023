import pickle

import dvc.api
import numpy as np
from skl2onnx import to_onnx
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main() -> None:

    data = dvc.api.read("data/iris_model.pkl", mode="rb")
    clf = pickle.loads(data)
    iris = load_iris()
    input, y = iris.data, iris.target
    input = input.astype(np.float32)
    X_train, _, _, _ = train_test_split(input, y)
    input = X_train.astype(np.float32)
    options = {id(clf): {"zipmap": False}}
    onx = to_onnx(clf, input, options=options)

    with open("model.onnx", "wb") as f:
        f.write(onx.SerializeToString())


if __name__ == "__main__":
    main()
