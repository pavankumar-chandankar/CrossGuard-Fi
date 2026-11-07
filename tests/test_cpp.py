"""Independent numerical parity, negative testing and 270-channel C++ checks."""
import csv
from io import StringIO
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cpp_executable(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler unavailable")
    path = tmp_path_factory.mktemp("cpp") / "csi_features"
    source = Path(__file__).resolve().parents[1] / "cpp/csi_features.cpp"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
                    "-Werror", str(source), "-o", str(path)], check=True)
    return path


def call_cpp(executable, rows, channels="270"):
    return subprocess.run([str(executable), channels], input=rows,
                          text=True, capture_output=True)


def test_cpp_numpy_parity_270_channels(cpp_executable):
    rng = np.random.default_rng(2026)
    values = rng.uniform(-2, 2, size=(100, 270))
    text = "\n".join(",".join(f"{v:.12g}" for v in row) for row in values) + "\n"
    run = call_cpp(cpp_executable, text)
    assert run.returncode == 0, run.stderr
    table = list(csv.DictReader(StringIO(run.stdout)))
    assert len(table) == 270
    assert np.allclose([float(row["mean"]) for row in table], values.mean(axis=0), atol=2e-6)
    assert np.allclose([float(row["stddev"]) for row in table],
                       values.std(axis=0), atol=2e-6)


@pytest.mark.parametrize("bad", ["", "1,2\n3\n", "1,nan\n", "1,inf\n",
                                   "1,2,3\n", "1,2,\n", "1,x\n", "1,2\n\n"])
def test_cpp_rejects_corrupt_input(cpp_executable, bad):
    run = call_cpp(cpp_executable, bad, "2")
    assert run.returncode != 0
    assert run.stderr


def test_cpp_rejects_invalid_channel_arg(cpp_executable):
    assert call_cpp(cpp_executable, "1,2\n", "0").returncode != 0
