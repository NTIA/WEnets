from pkg_resources import resource_filename

from wawenets import modeselektor
from wawenets.inference import Predictor


class TestPredictor:
    def test_predict(self):
        config = modeselektor[1]
        predictor = Predictor(**config)
        wav_path = resource_filename("speech", "T000093_Q446_D401.wav")
        result = predictor.predict(wav_path)
        assert result == 2.186994658410549

    def test_denormalize(self):
        # TODO: do pytest right and only make the predictor once
        config = modeselektor[1]
        predictor = Predictor(**config)
        denormalized = predictor.denormalize(0, (0, 2))
        assert denormalized == 1
