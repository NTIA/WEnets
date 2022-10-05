from typing import Tuple

import torch

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
        "multitarget_obj_2022": {
            "model_class": WAWEnet2020,
            "num_targets": 7,
            "channels": 96,
        },
        "multitarget_subj_obj_2022": {
            "model_class": WAWEnet2020,
            "num_targets": 11,
            "channels": 96,
        },
    }

    def __init__(
        self,
        *args,
        model_type: str = "multitarget2022",
        weights_path: str = "",
        normalization_ranges: list = [],
        predictor_names: list = [],
        **kwargs,
    ) -> None:
        """hi"""
        self.params = self.model_params[model_type]
        self.model = self.params["model_class"](**self.params)
        if not weights_path:
            raise ValueError("no model weight specified, invalid configuration")
        if not normalization_ranges:
            raise ValueError("no normalization ranges specified, invalid configuration")
        self.normalization_ranges = normalization_ranges
        if not predictor_names:
            raise ValueError("no predictor names specified, invalid configuration")
        self.predictor_names = predictor_names
        weights = torch.load(weights_path)
        self.model.load_state_dict(weights["model_state_dict"])
        self.model.eval()

    def predict(self, audio_tensor: torch.tensor):
        """makes a prediction on specified batch of audio tensors"""
        with torch.no_grad():
            prediction = self.model(audio_tensor).detach().numpy()

        # denormalize all predictions
        normalized_predictions = list()
        for prediction_ind in range(prediction.shape[0]):
            current_prediction = prediction[prediction_ind]
            assert len(current_prediction) == len(self.normalization_ranges)
            current_prediction = {
                name: self._denormalize(pred, norm_range)
                for pred, norm_range, name in zip(
                    current_prediction,
                    self.normalization_ranges,
                    self.predictor_names,
                )
            }
            normalized_predictions.append(current_prediction)

        return normalized_predictions

    def _denormalize(self, prediction, target_minmax: Tuple):
        """converts a normalized target to the equivalent value in
        [target_min, target_max]"""
        # this is a fact about our model, maybe this should be in the config
        prediction_min, prediction_max = -1, 1
        # target range varies by target type
        target_min, target_max = target_minmax

        old_range = prediction_max - prediction_min
        target_range = target_max - target_min
        return (((prediction - prediction_min) * target_range) / old_range) + target_min
