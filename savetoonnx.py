from pathlib import Path

import dvc.api
import hydra
import mlflow
import mlflow.onnx
import pandas as pd
from hydra.core.config_store import ConfigStore
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
from sklearn.model_selection import train_test_split
import numpy as np
from conf.config import Params
from iris import iris_classifier
from sklearn.datasets import load_iris
import pickle


def main() -> None:

    data = dvc.api.read("data/iris_model.pkl", mode='rb')
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
