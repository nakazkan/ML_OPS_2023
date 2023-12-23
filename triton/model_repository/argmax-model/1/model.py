import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            probs = pb_utils.get_input_tensor_by_name(request, "probabilities").as_numpy()
            labels = np.argmax(probs)
            output_labels = pb_utils.Tensor("labels", np.array(labels))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_labels]
            )
            responses.append(inference_response)
        return responses