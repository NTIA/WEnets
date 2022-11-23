from typing import List, Tuple

import torch
import torch.nn as nn


def conv_section(
    channels: int,
    pad_spec: Tuple[int, int],
    activation: str,
    pool_kernel_size: int,
    pool_method: str,
    first_section: bool = False,
) -> List[nn.Module]:
    """
    creates a WAWEnet-style model "section" containing a convolution, a
    batchnorm, an activation function, and a pooling layer.

    Parameters
    ----------
    channels : int
        the number of channels the convolution section should have
    pad_spec : Tuple[int, int]
        the amount to pad the beginning and end of the audio tensor
    activation : str
        the type of activation function to use, either "relu" or "prelu"
    pool_kernel_size : int
        the size of the pooling kernel
    pool_method : str
        the pooling method to be used, either "avg" or "max"
    first_section : bool, optional
        whether or not this is the first section in the model. if true, the
        number of input channels is set to 1, by default False

    Returns
    -------
    List[nn.Module]
        a list of PyTorch modules, suitable for input to `nn.Sequential`

    Raises
    ------
    ValueError
        if an unsupported pooling or activation function is specified.
    """
    # set activation function for this section
    if activation == "prelu":
        activation_function = nn.PReLU(channels)
    elif activation == "relu":
        activation_function = nn.ReLU()
    else:
        raise ValueError(f"unsupported activation function: {activation}")
    # set pooling method for this section
    if pool_method == "avg":
        pool_function = nn.AvgPool1d
    elif pool_method == "max":
        pool_function = nn.MaxPool1d
    else:
        raise ValueError(f"unsupported pooling method: {pool_method}")
    layers = list()
    # add a padding layer if needed
    if any(pad_spec):
        layers.extend([nn.ConstantPad1d((pad_spec[0], pad_spec[1]), 0)])
    # special case if this is the first section, only one channel input
    if first_section:
        channels_in = 1
    else:
        channels_in = channels
    channels_out = channels
    layers.extend(
        [
            nn.Conv1d(channels_in, channels_out, 3, padding=1),
            nn.BatchNorm1d(channels_out),
            activation_function,
            pool_function(pool_kernel_size),
        ]
    )
    return layers


def create_features(
    channels: int, features_spec: List[Tuple[tuple, str, int, str]]
) -> nn.Sequential:
    """
    creates the feature extractor for a WAWEnet model.

    Parameters
    ----------
    channels : int
        the number of channels each convolutional layer should use
    features_spec : List[Tuple[tuple, str, int, str]]
        a list of tuples containing the configuration for each section. the
        expected format for each tuple is:
        ((pad_start, pad_end), "activation", pool_kernel_size, "pool_type")

    Returns
    -------
    nn.Sequential
        a WAWEnet feature extractor
    """
    modules = list()
    # first section is a special case
    modules.extend(conv_section(channels, *features_spec[0], first_section=True))
    for section in features_spec[1:]:
        modules.extend(conv_section(channels, *section))
    return nn.Sequential(*modules)


class WAWEnetICASSP2020(nn.Module):
    """
    generates a WAWEnet-style model like the one reported in "WAWEnets: A
    No-Reference Convolutional Waveform-Based Approach to Estimating Narrowband
    and Wideband Speech Quality", published at ICASSP 2020
    """

    features_spec = [
        ((0, 0), "prelu", 2, "avg"),
        ((0, 0), "prelu", 4, "max"),
        ((0, 0), "prelu", 2, "max"),
        ((0, 0), "prelu", 4, "max"),
        ((0, 0), "prelu", 3, "max"),
        ((0, 0), "prelu", 2, "max"),
        ((1, 2), "prelu", 2, "max"),
        ((0, 0), "prelu", 2, "max"),
        ((0, 0), "prelu", 32, "avg"),
    ]

    def __init__(self, *args, num_targets: int = 1, channels: int = 96, **kwargs):
        """
        initializes a WAWEnet by creating the feature extractor and mapper.

        Parameters
        ----------
        num_targets : int, optional
            the number of targets to be predicted by the WAWEnet, by default 1
        channels : int, optional
            the number of channels each convolutional layer should use, by
            default 96
        """
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.features = create_features(channels, self.features_spec)
        self.mapper = nn.Sequential(
            # 1 sample, 96 "features"
            nn.Linear(96, num_targets),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs inference on `x`

        Parameters
        ----------
        x : torch.Tensor
            audio input data normalized to [-1, 1] with dimensions
            [batch, 1, 48,000]

        Returns
        -------
        torch.Tensor
            WAWEnet predictions with dimention
            [batch, num_targets]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.mapper(x)
        return x

    def _initialize_weights(self):
        """
        initializes weights in the WAWEnet model using the Kaiming normal
        method
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)


class WAWEnet2020(nn.Module):
    """
    generates a WAWEnet-style model like the one reported in "Wideband
    Audio Waveform Evaluation Networks: Efficient, Accurate Estimation
    of Speech Qualities", published on arXiv: https://arxiv.org/abs/2206.13272
    """

    features_spec = [
        # 48,000 samples
        # filter size ~= 0.188 ms, equiv sample rate = 16000 hz, or 0.0625 ms
        ((0, 0), "relu", 4, "avg"),
        # 12,000 samples
        # filter size ~= 0.188 ms, equiv sample rate = 4000 hz, or 0.25 ms
        ((0, 0), "relu", 2, "avg"),
        # 6,000 samples
        # filter size ~= 1.5 ms, equiv sample rate = 2000 hz, or 0.5 ms
        ((0, 0), "relu", 2, "avg"),
        # 3,000 samples
        # filter size ~= 3 ms, equiv sample rate = 1000 hz, or 1 ms
        ((0, 0), "relu", 4, "avg"),
        # 750 samples
        # filter size ~= 12 ms, equiv sample rate = 250 hz, or 4 ms
        ((0, 0), "relu", 2, "avg"),
        # 375 samples
        # filter size ~= 24 ms, equiv sample rate = 125, or 8 ms
        ((0, 1), "relu", 2, "avg"),
        # 188 samples
        # filter size ~= 48 ms, equiv sample rate = 62.5 hz, or 16 ms
        ((0, 0), "relu", 2, "avg"),
        # 94 samples
        # filter size ~= 96 ms, equiv sample rate = 31.25 hz, or 32 ms
        ((0, 0), "relu", 2, "avg"),
        # 47 samples
        # filter size ~= 192 ms, equiv sample rate = 15.625 hz, or 64 ms
        ((0, 1), "relu", 2, "avg"),
        # 24 samples
        # filter size ~= 384 ms, equiv sample rate = 7.8125 hz, or 128 ms
        ((0, 0), "relu", 2, "avg"),
        # 12 samples
        # filter size ~= 768 ms, equiv sample rate = 3.90625 hz, or 256 ms
        ((0, 0), "relu", 2, "avg"),
        # 6 samples
        # filter size ~= 1536 ms, equiv sample rate = 1.953125 hz, or 512 ms
        ((0, 0), "relu", 2, "avg"),
        # 3 samples
        # filter size ~= 3.1 sec, equiv sample rate = 0.977 hz, or 1024 ms
        ((0, 0), "relu", 3, "avg"),
        # only one sample left!
    ]

    def __init__(self, *args, num_targets: int = 1, channels: int = 96, **kwargs):
        """
        initializes a WAWEnet by creating the feature extractor and mapper.

        Parameters
        ----------
        num_targets : int, optional
            the number of targets to be predicted by the WAWEnet, by default 1
        channels : int, optional
            the number of channels each convolutional layer should use, by
            default 96
        """
        super().__init__()
        self.channels = channels
        self.features = create_features(channels, self.features_spec)
        self.mapper = nn.Sequential(
            # 1 sample, 96 "features"
            nn.Linear(96, num_targets),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs inference on `x`

        Parameters
        ----------
        x : torch.Tensor
            audio input data normalized to [-1, 1] with dimensions
            [batch, 1, 48,000]

        Returns
        -------
        torch.Tensor
            WAWEnet predictions with dimention
            [batch, num_targets]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.mapper(x)
        return x

    def _initialize_weights(self):
        """
        initializes weights in the WAWEnet model using the Kaiming normal
        method
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)
