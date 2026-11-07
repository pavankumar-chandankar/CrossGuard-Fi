# CrossGuard-Fi — Wi-Fi CSI feature extraction and inference interface

**Scope:** A small C++17/Python prototype for processing headerless Wi-Fi channel-state-information (CSI) amplitude windows. The existing C++ extractor computes a mean and population standard deviation for each channel. A Python CLI now validates input, invokes the C++ program, and can optionally apply a user-supplied PyTorch checkpoint.

This is **not** a complete Wi-Fi sensing or activity recognition deployment. No CSI capture driver, training pipeline, trained model checkpoint, accuracy benchmark, or real-time inference service is provided.

## Project layout

- `cpp/csi_features.cpp`: deterministic streaming statistics, with finite-value and consistent-channel checks
- `python/pipeline.py`: CSI CSV validation, subprocess invocation and optional classifier inference
- `tests/test_crossguard.py`: numeric regression, malformed-input and end-to-end CLI checks
- `.github/workflows/tests.yml`: GitHub Actions regression tests and Clang static analysis

## Reproduce the features

Requires Python 3 and a C++17 compiler:

```bash
g++ -std=c++17 -Wall -Wextra -Werror cpp/csi_features.cpp -o csi_features
printf '1,2\n3,4\n' > window.csv
./csi_features 2 < window.csv
python3 python/pipeline.py --csv window.csv --extractor ./csi_features
python3 -m unittest discover -s tests -v
```

For the example above, channel 0 has mean 2 and standard deviation 1; channel 1 has mean 3 and standard deviation 1. The Python CLI returns a JSON payload containing four numerical features and `prediction: null` when no checkpoint is provided.

## Optional PyTorch inference

Install PyTorch separately and specify `--checkpoint path/to/model.pt`. The checkpoint must contain `hidden_dim`, `num_classes`, and `state_dict` compatible with `Sequential(Linear(2 * channels, hidden_dim), ReLU(), Linear(hidden_dim, num_classes))`. Checkpoints load using `weights_only=True`. An actual classifier must first be trained and evaluated on appropriate data; no model or recognition performance is claimed here.

## Input and validation

The input is a CSV file **without a header**. Each row represents one sample in the window and each comma-separated column one CSI amplitude channel. Empty windows, inconsistent channel counts, nonnumeric values, and NaNs/infinities are rejected. Complex-valued CSI, phase correction and channel calibration are outside this prototype's scope.

## CV evidence

This repository supports claims of **C++17 CSI amplitude feature extraction, Python input validation, a checkpoint-based inference interface, and automated regression tests**. It does not establish deployed HAR performance, trained model accuracy, or a production Wi-Fi pipeline.
