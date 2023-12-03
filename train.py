from pathlib import Path

import dvc.api
import hydra
import mlflow
import mlflow.onnx
import pandas as pd
from hydra.core.config_store import ConfigStore
from matplotlib import pyplot as plt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from conf.config import Params
from iris import iris_classifier

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


def save_get_scatter_path(artifact_path):
    plot_path = Path(artifact_path)
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path)
    return plot_path


def save_plots(clf, X_val, y_val, y_train, data_path, n_classes=3):
    y_score = clf.predict_proba(X_val)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_val)
    paths = []
    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, i],
            y_score[:, i],
            name=f"{i} vs the rest",
            color="darkorange",
        )
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"One-vs-Rest ROC curves:\n {i} class vs All ")
        plt.legend()
        result_path = data_path + "plots/" + f"roc_auc_{i}_label.png"
        paths.append(result_path)
        save_get_scatter_path(result_path)
    return paths


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    with dvc.api.open(
        "data/train.csv", repo="https://github.com/nakazkan/ML_OPS_2023"
    ) as f:
        df = pd.read_csv(f)

    target = df[df.columns[0]]
    X = df.drop(columns=[df.columns[0]])
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, test_size=cfg.data.val_size, random_state=cfg.data.seed
    )
    clf = iris_classifier.IrisClassifier(**cfg.model)
    clf.fit(X_train, y_train)

    output_file = Path(cfg.data.path + "iris_model.pkl")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    clf.save_model(cfg.data.path + "iris_model.pkl")

    onnx_model = convert_sklearn(
        clf.clf, initial_types=[("input", FloatTensorType(X_val.shape))]
    )

    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    precision = precision_score(y_val, y_pred, average="macro")
    recall = recall_score(y_val, y_pred, average="macro")
    plot_paths = save_plots(clf, X_val, y_val, y_train, cfg.data.path, n_classes=3)

    mlflow.set_tracking_uri(uri=cfg.mlflow.uri)
    mlflow.set_experiment("Siple iris dataset experiment")

    with mlflow.start_run():
        mlflow.log_params(cfg.model)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1 macro", f1)
        mlflow.log_metric("precison macro", precision)
        mlflow.log_metric("recall macro", recall)

        [mlflow.log_artifact(plot) for plot in plot_paths]

        mlflow.set_tag("Training Info", "Basic Random Forest model for iris data")

        mlflow.onnx.log_model(onnx_model, "model")


if __name__ == "__main__":
    main()
