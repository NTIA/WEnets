import os

from pathlib import Path
from pkg_resources import resource_filename

import yaml


def training_params(training_regime: str = "default") -> dict:
    """
    reads a config file and returns a config dictionary.

    Parameters
    ----------
    training_regime : str
        the name of the training regime in use

    Returns
    -------
    dict
        a dictionary containing params required to train the given regime
    """
    train_config_file = resource_filename("config", "train_config.yaml")
    with open(train_config_file) as fp:
        train_config = yaml.safe_load(fp.read())
    config = train_config["training_regimes"][training_regime]
    config["output_uri"] = get_results_path()
    return config


# TODO: get the user dir and make a place to put the training results
def get_results_path():
    user_home_path = Path(os.path.expanduser("~"))
    results_path = user_home_path / "wenets_training_artifacts"
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path
