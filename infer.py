import dvc.api
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.metrics import accuracy_score

from conf.config import Params
from iris import iris_classifier

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:

    with dvc.api.open("data/test.csv") as f:
        df = pd.read_csv(f)

    clf = iris_classifier.IrisClassifier(
        cfg.model.random_state, cfg.model.n_estimators, cfg.model.max_depth
    )
    clf = clf.load_model(cfg.data.path + "iris_model.pkl")

    y_test = df[df.columns[0]]
    X_test = df.drop(columns=[df.columns[0]])

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    out = pd.DataFrame(y_pred)
    out.to_csv(cfg.data.path + "out.csv")


if __name__ == "__main__":
    main()
