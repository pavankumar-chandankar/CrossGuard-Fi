import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FeatureRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tempdir.name) / "csi_features"
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise unittest.SkipTest("Requires a C++17 compiler")
        subprocess.run(
            [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
             str(ROOT / "cpp/csi_features.cpp"), "-o", str(cls.binary)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tempdir"):
            cls.tempdir.cleanup()

    def test_known_mean_and_population_stddev(self):
        result = subprocess.run(
            [str(self.binary), "2"], input="1,2\n3,4\n",
            text=True, capture_output=True, check=True,
        )
        self.assertIn("0,2.000000,1.000000", result.stdout)
        self.assertIn("1,3.000000,1.000000", result.stdout)

    def test_rejects_malformed_or_nonfinite(self):
        for content in ("", "1,2\n3\n", "1,nan\n"):
            result = subprocess.run(
                [str(self.binary), "2"], input=content,
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0, content)

    def test_python_feature_pipeline(self):
        path = Path(self.tempdir.name) / "example.csv"
        path.write_text("1,2\n3,4\n", encoding="utf-8")
        output = subprocess.run(
            [sys.executable, str(ROOT / "python/pipeline.py"),
             "--csv", str(path), "--extractor", str(self.binary)],
            check=True, text=True, capture_output=True,
        )
        payload = json.loads(output.stdout)
        self.assertEqual(payload["features"], [2.0, 3.0, 1.0, 1.0])
        self.assertIsNone(payload["prediction"])


if __name__ == "__main__":
    unittest.main()
