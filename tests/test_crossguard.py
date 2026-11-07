"""Compatibility checks for the legacy C++/Python CLI, retained from initial project scope."""
from pathlib import Path
import json
import subprocess
import sys


def test_legacy_interface(tmp_path, cpp_executable=None):
    # Build in the test itself to avoid depending on a separate fixture module.
    from shutil import which
    import pytest
    compiler = which("g++") or which("clang++")
    if not compiler:
        pytest.skip("No C++ compiler")
    source = Path(__file__).resolve().parents[1] / "cpp/csi_features.cpp"
    executable = tmp_path / "csi_features"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(executable)], check=True)
    csv_file = tmp_path / "demo.csv"
    csv_file.write_text("1,2\n3,4\n", encoding="utf-8")
    cli = Path(__file__).resolve().parents[1] / "python/pipeline.py"
    run = subprocess.run([sys.executable, str(cli), "--csv", str(csv_file),
                          "--extractor", str(executable)], capture_output=True,
                         text=True, check=True)
    assert json.loads(run.stdout)["features"] == [2.0, 3.0, 1.0, 1.0]
