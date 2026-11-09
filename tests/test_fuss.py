import numpy as np
import pytest

from dotears.errors import ConfigurationError, MediaError
from dotears.fuss import LocalFUSS


def test_unprepared_fuss_graph_is_rejected_before_optional_runtime_import(tmp_path):
    with pytest.raises(ConfigurationError, match="artifact"):
        LocalFUSS(tmp_path)


def test_fuss_adapter_checks_shape_finiteness_and_mixture_consistency():
    model = LocalFUSS.__new__(LocalFUSS)
    model.input, model.output = "input", "output"

    class Session:
        def run(self, _output, feed):
            values = feed["input"][0, 0]
            return np.stack([values / 4] * 4)[None]

    model.session = Session()
    audio = np.linspace(-0.1, 0.1, 48000, dtype=np.float32)
    assert np.array_equal(model.separate(audio).sum(0), audio)
    with pytest.raises(ConfigurationError):
        model.separate(np.full(48000, np.nan))

    class Broken:
        def run(self, _output, feed):
            return np.ones((1, 4, 48000), dtype=np.float32)

    model.session = Broken()
    with pytest.raises(MediaError):
        model.separate(audio)
