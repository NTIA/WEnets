from pathlib import Path
from typing import Tuple

import torch

from wawenets.data import load_audio_to_tensor
from wawenets.model import WAWEnetICASSP2020, WAWEnet2020


class Predictor:
    """wraps model creation and data loading"""

    model_params = {
        "icassp2020": {
            "model_class": WAWEnetICASSP2020,
            "num_targets": 1,
            "channels": 96,
        },
        "wawenet2020": {"model_class": WAWEnet2020, "num_targets": 1, "channels": 96},
        "multitarget2022": {
            "model_class": WAWEnet2020,
            "num_targets": 12,
            "channels": 96,
        },
    }

    def __init__(
        self,
        model_type: str = "multitarget2022",
        weights_path: str = "",
        normalization_ranges: list = [],
    ) -> None:
        """hi"""
        self.params = self.model_params[model_type]
        self.model = self.params["model_class"](**self.params)
        if not weights_path:
            raise ValueError("no model weight specified, invalid configuration")
        if not normalization_ranges:
            raise ValueError("no normalization ranges specified, invalid configuration")
        self.normalization_ranges = normalization_ranges
        weights = torch.load(weights_path)
        self.model.load_state_dict(weights["model_state_dict"])

    def predict(self, file_path: Path):
        """loads a file, makes a prediction"""
        audio_tensor = load_audio_to_tensor(file_path)
        prediction = self.model(audio_tensor).detach().numpy()
        if prediction.size > 1:
            prediction = list(prediction[0])
            assert len(prediction) == len(self.normalization_ranges)
            prediction = [
                self.denormalize(pred, norm_range)
                for pred, norm_range in zip(prediction, self.normalization_ranges)
            ]
        else:
            prediction = self.denormalize(
                prediction.item(), self.normalization_ranges[0]
            )
        return prediction

    def denormalize(self, prediction, target_minmax: Tuple):
        """converts a normalized target to the equivalent value in
        [target_min, target_max]"""
        # fact about our model, maybe this should be in the config
        prediction_min, prediction_max = -1, 1
        # target range varies by target type
        target_min, target_max = target_minmax

        old_range = prediction_max - prediction_min
        target_range = target_max - target_min
        return (((prediction - prediction_min) * target_range) / old_range) + target_min
