from pkg_resources import resource_filename

__version__ = "0.1.0"

# TODO: maybe this config-type stuff should live in a yaml somewhere?
normalization_ranges = {
    "PESQMOSLQO": (1.01, 4.64),
    "POLQAMOSLQO": (1, 4.75),
    "PEMO": (0, 1),
    "VISQOL": (1, 5),
    "STOI": (0.45, 1),
    "ESTOI": (0.23, 1),
    "SIIBGAUSS": (0, 750),
}

modeselektor = {
    1: {
        "model_type": "wawenet2020",
        "weights_path": resource_filename(
            "weights", "20200801_WAWEnetFD13FC96AvgReLU_PESQMOSLQO.pt"
        ),
        "normalization_ranges": [normalization_ranges["PESQMOSLQO"]],
    },
    2: {
        "model_type": "wawenet2020",
        "weights_path": resource_filename(
            "weights", "20200801_WAWEnetFD13FC96AvgReLU_POLQAMOSLQO_.pt"
        ),
        "normalization_ranges": [normalization_ranges["POLQAMOSLQO"]],
    },
    3: {
        "model_type": "wawenet2020",
        "weights_path": resource_filename(
            "weights", "20200802_WAWEnetFD13FC96AvgReLU_PEMO.pt"
        ),
        "normalization_ranges": [normalization_ranges["PEMO"]],
    },
    4: {
        "model_type": "wawenet2020",
        "weights_path": resource_filename(
            "weights", "20200802_WAWEnetFD13FC96AvgReLU_STOI.pt"
        ),
        "normalization_ranges": [normalization_ranges["STOI"]],
    },
    5: {
        "model_type": "multitarget2022",
        "weights_path": resource_filename("weights", ""),
        # order matters
        "normalization_ranges": [],
    },
}
