from dataclasses import asdict
import numpy as np
import pytest

from crossguard_fi.config import CONTRACT, CSIContract
from crossguard_fi.data.validation import validate_csi, DataValidationError


def sample():
    csi = np.zeros((100, CONTRACT.channels), dtype=np.float32)
    ts = np.arange(100, dtype=np.int64) * 20_000_000
    return csi, ts


def test_fixed_contract():
    assert (CONTRACT.channels, CONTRACT.sampling_hz, CONTRACT.classes) == (270, 50, 24)
    assert asdict(CONTRACT)["window_samples"] == 100


def test_valid_signal():
    csi, ts = sample()
    assert validate_csi(csi, ts).samples == 100


@pytest.mark.parametrize("mutation", ["shape", "nan", "infinity", "large", "duplicate_ts", "gap", "short"])
def test_reject_malformed_signal(mutation):
    csi, ts = sample()
    if mutation == "shape":
        csi = csi[:, :-1]
    elif mutation == "nan":
        csi[0, 0] = np.nan
    elif mutation == "infinity":
        csi[0, 0] = np.inf
    elif mutation == "large":
        csi[0, 0] = 1e9
    elif mutation == "duplicate_ts":
        ts[1] = ts[0]
    elif mutation == "gap":
        ts[1:] += 200_000_000
    elif mutation == "short":
        csi, ts = csi[:-1], ts[:-1]
    with pytest.raises(DataValidationError):
        validate_csi(csi, ts)
