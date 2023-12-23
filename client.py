from functools import lru_cache

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_labels(data: np.ndarray):
    triton_client = get_client()
    text = np.array([data], dtype=np.float32).reshape(4)

    input_data = InferInput(
        name="X", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_data.set_data_from_numpy(text)

    infer_output = InferRequestedOutput("labels")
    query_response = triton_client.infer(
        "ensemble-onnx", [input_data], outputs=[infer_output]
    )
    labels = query_response.as_numpy("labels")
    return labels


def call_triton_probs(data: np.ndarray):
    triton_client = get_client()
    text = np.array([data], dtype=np.float32)
    input_data = InferInput(
        name="X", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_data.set_data_from_numpy(text)
    query_response = triton_client.infer(
        "onnx-model", [input_data], outputs=[InferRequestedOutput("probabilities")]
    )
    probs = query_response.as_numpy("probabilities")[0]
    return probs


def main():

    data = [[0.0, 0.0, 0.0, 0.0], [5.7, 2.8, 4.5, 1.3], [6.2, 3.0, 5.0, 2.0]]

    for i in range(3):
        labels = call_triton_labels(data[i])
        assert (labels == np.array([i])).all()

    for i in range(3):
        probs = call_triton_probs(data[i])
        arg_max = max(probs)
        assert arg_max > 0.9

    print("All tests passed")


if __name__ == "__main__":
    main()
