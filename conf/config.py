from dataclasses import dataclass


@dataclass
class Data:
    path: str
    seed: int
    test_size: float


@dataclass
class Model:
    seed: int
    n_estimators: int
    max_depth: int


@dataclass
class Params:
    data: Data
    model: Model
