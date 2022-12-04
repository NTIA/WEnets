import os
import yaml

from collections import OrderedDict
from pathlib import Path
from pkg_resources import resource_filename

__version__ = "0.1.0"

# future work: maybe this config-type stuff should live in a yaml somewhere?

# use these ranges to map WAWEnet outputs to prediction-type ranges
normalization_ranges = OrderedDict(
    {
        "mos": (1, 5),
        "noi": (1, 5),
        "col": (1, 5),
        "dis": (1, 5),
        "PESQMOSLQO": (1.01, 4.64),
        "POLQAMOSLQO": (1, 4.75),
        "PEMO": (0, 1),
        "VISQOL": (1, 5),
        "STOI": (0.45, 1),
        "ESTOI": (0.23, 1),
        "SIIBGAUSS": (0, 750),
    }
)

# configuration for all the different WAWEnets
modeselektor = {
    1: {
        "model_type": "wawenet2020",
        "weights_path": resource_filename(
            "weights", "20200801_WAWEnetFD13FC96AvgReLU_PESQMOSLQO.pt"
        ),
        "normalization_ranges": [normalization_ranges["PESQMOSLQO"]],
        "predictor_names": ["PESQMOSLQO"],
    },
    2: {
        "model_type": "wawenet2020",
        "weights_path": resource_filename(
            "weights", "20200801_WAWEnetFD13FC96AvgReLU_POLQAMOSLQO_.pt"
        ),
        "normalization_ranges": [normalization_ranges["POLQAMOSLQO"]],
        "predictor_names": ["POLQAMOSLQO"],
    },
    3: {
        "model_type": "wawenet2020",
        "weights_path": resource_filename(
            "weights", "20200802_WAWEnetFD13FC96AvgReLU_PEMO.pt"
        ),
        "normalization_ranges": [normalization_ranges["PEMO"]],
        "predictor_names": ["PEMO"],
    },
    4: {
        "model_type": "wawenet2020",
        "weights_path": resource_filename(
            "weights", "20200802_WAWEnetFD13FC96AvgReLU_STOI.pt"
        ),
        "normalization_ranges": [normalization_ranges["STOI"]],
        "predictor_names": ["STOI"],
    },
    5: {
        "model_type": "multitarget_obj_2022",
        "weights_path": resource_filename(
            "weights",
            "POLQAMOSLQO_PESQMOSLQO_STOI_PEMO_ViSQOL3_c310_ESTOI_SIIBGauss_5d8c59ea6bc24b019b4fe6ac5b87d3db.pt",
        ),
        # order matters
        "normalization_ranges": list(normalization_ranges.values())[4:],
        "predictor_names": list(normalization_ranges.keys())[4:],
    },
    6: {
        "model_type": "multitarget_subj_obj_2022",
        "weights_path": resource_filename(
            "weights",
            "mos_noi_col_dis_PESQMOSLQO_POLQAMOSLQO_PEMO_ViSQOL3_C310_STOI_ESTOI_SIIBGauss_5dd8fccf49a84289b9260524ac38ccbe.pt",
        ),
        # order matters
        "normalization_ranges": list(normalization_ranges.values()),
        "predictor_names": list(normalization_ranges.keys()),
    },
}


def get_stl_path() -> str:
    """
    returns the path to the STL bin dir based on the contents of config.yaml

    Returns
    -------
    str
        path to the location of `config.yaml` as a string

    Raises
    ------
    FileNotFoundError
        if `config.yaml` is not found in the expected place
    """
    current_path = Path(os.path.realpath(__file__))
    config_path = current_path.parent.parent / "config" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"unable to find `config.yaml` in {config_path}. please follow the setup "
            "instructions in README.md to create `config.yaml"
        )
    with open(config_path) as yaml_fp:
        config = yaml.safe_load(yaml_fp)
    return config["StlBinPath"]
