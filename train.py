from pathlib import Path

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from conf.config import Params
from iris import iris_classifier

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=cfg.data.seed, test_size=cfg.data.test_size
    )
    df_test = pd.DataFrame(X_test, y_test)
    df_train = pd.DataFrame(X_train, y_train)

    clf = iris_classifier.IrisClassifier()
    clf.fit(X_train, y_train)

    output_file = Path(cfg.data.path + "iris_model.pkl")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    clf.save_model(cfg.data.path + "iris_model.pkl")

    df_test.to_csv(cfg.data.path + "test.csv")
    df_train.to_csv(cfg.data.path + "train.csv")


if __name__ == "__main__":
    main()
