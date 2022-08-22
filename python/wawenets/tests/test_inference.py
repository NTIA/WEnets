import os
import yaml

from pathlib import Path
from pkg_resources import resource_filename

import numpy as np

from wawenets import modeselektor
from wawenets.data import WavHandler
from wawenets.inference import Predictor

import pytest


@pytest.fixture
def stl_path():
    test_path = Path(os.path.realpath(__file__))
    config_path = test_path.parent.parent / "config.yaml"
    with open(config_path) as yaml_fp:
        config = yaml.safe_load(yaml_fp)
    return config["StlBinPath"]


@pytest.fixture
def predictor():
    config = modeselektor[1]
    predictor = Predictor(**config)
    return predictor


@pytest.fixture
def wav_path():
    return Path(resource_filename("speech", "T000093_Q446_D401.wav"))


class TestPredictor:
    def test_predict(self, predictor, wav_path, stl_path):
        with WavHandler(wav_path, True, stl_path) as wh:
            prepared_tensor = wh.prepare_tensor()
            result = predictor.predict(prepared_tensor)
            assert np.isclose(result, 2.18705556511879)

    def test_denormalize(self, predictor):
        denormalized = predictor.denormalize(0, (0, 2))
        assert denormalized == 1
