from pathlib import Path

from essentia.standard import TensorflowPredictMusiCNN
import numpy as np


class algoTensorflowPredictMusiCNN:
    def __init__(self, kwargs, audio_duration=60, models_base=None):
        self.kwargs = kwargs
        self.model = None
        self.sample_rate = 16000
        self.data = np.ones(self.sample_rate * audio_duration, dtype="float32")

        if models_base is not None:
            self.kwargs["graphFilename"] = str(
                Path(models_base, self.kwargs["graphFilename"])
            )

    def instantiate(self, warmup_data=None):
        self.model = TensorflowPredictMusiCNN(**self.kwargs)
        if warmup_data:
            self.model(warmup_data)

    def inference(self):
        assert (
            self.model is not None
        ), "The model has to be instantiated via `self.instantiate()`"

        self.model(self.data)
