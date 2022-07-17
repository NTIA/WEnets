from typing import Tuple


import torch.nn as nn

# handle model stuff here


def conv_section(
    channels: int,
    pad_spec: Tuple[int, int],
    activation: str,
    pool_kernel_size: int,
    pool_method: str,
    first_section: bool = False,
):
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


def create_features(channels, features_spec) -> nn.Sequential:
    modules = list()
    # first section is a special case
    modules.extend(conv_section(channels, *features_spec[0], first_section=True))
    for section in features_spec[1:]:
        modules.extend(conv_section(channels, *section))
    return nn.Sequential(*modules)


class WAWEnet2020(nn.Module):
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

    def __init__(self, num_targets: int = 1, channels: int = 96):
        super().__init__()
        self.channels = channels
        self.features = create_features(channels, self.features_spec)
        self.mapper = nn.Sequential(
            # 1 sample, 96 "features"
            nn.Linear(96, num_targets),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.mapper(x)
        return x

    def _initialize_weights(self):
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


class WAWEnet2022(nn.Module):
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

    def __init__(self, num_targets: int = 1, channels: int = 96):
        super().__init__()
        self.channels = channels
        self.features = create_features(channels, self.features_spec)
        self.mapper = nn.Sequential(
            # 1 sample, 96 "features"
            nn.Linear(96, num_targets),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.mapper(x)
        return x

    def _initialize_weights(self):
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
