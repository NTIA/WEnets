import logging
from pathlib import Path

import fairseq
import torch
import torch.nn as nn
from wawenets.stl_wrapper import call_command

logging.getLogger("fairseq").setLevel(logging.WARNING)


class Wav2VecRef(nn.Module):
    def __init__(
        self,
        weights_path: str = None,
        wav2vec_features: int = 768,
        num_targets: int = 1,
        train_all: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if weights_path:
            self.weights_path = weights_path
        else:
            # download some weights
            self.weights_path = Path("wav2vec_small.pt")
            if self.weights_path.exists():
                self.weights_path = str(self.weights_path)
            else:
                print("downloading wav2vec pretrained weights")
                curl_args = [
                    "curl",
                    "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
                    "-o",
                    str(self.weights_path),
                ]
                std_out, std_err = call_command(
                    curl_args
                )  # need to handle out and err somehow, looks like everything gets shoved into err :/

        model, config, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(self.weights_path)]
        )
        self.model = model[0]
        self.model.remove_pretraining_modules()
        # only train the linear layer
        if not train_all:
            for params in self.model.parameters():
                params.requires_grad = False
        self.mapper = nn.Linear(wav2vec_features, num_targets)

    def forward(self, x: torch.Tensor):
        x = x.squeeze(1)  # TODO: may be necessary?
        x = self.model(x, mask=False, features_only=True)
        x = x["x"]
        x = torch.mean(x, 1)
        x = self.mapper(x)
        return x
