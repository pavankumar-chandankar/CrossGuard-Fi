from pathlib import Path
import pytest

from crossguard_fi.data.synthetic import generate_fixture


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("crossguard-fi")
    manifest = generate_fixture(root, subjects=3, trials_per_class=1,
                                samples=100, seed=41)
    return root, manifest
