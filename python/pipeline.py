#!/usr/bin/env python3
"""Validate CSI amplitude windows, run the C++ extractor, optionally infer with PyTorch."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess


def read_window(path):
    content = Path(path).read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(content)))
    if not rows or not rows[0]:
        raise ValueError("CSI window must contain numeric rows")
    width = len(rows[0])
    for line_number, row in enumerate(rows, 1):
        if len(row) != width:
            raise ValueError(f"line {line_number}: inconsistent channel count")
        for cell in row:
            try:
                value = float(cell)
            except ValueError as exc:
                raise ValueError(f"line {line_number}: nonnumeric CSI amplitude") from exc
            if not math.isfinite(value):
                raise ValueError(f"line {line_number}: non-finite CSI amplitude")
    return content, width, len(rows)


def extract_features(executable, content, channels):
    result = subprocess.run(
        [str(Path(executable).resolve()), str(channels)], input=content,
        text=True, capture_output=True, check=True,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(rows) != channels:
        raise RuntimeError("C++ extractor returned an unexpected number of rows")
    return [float(row["mean"]) for row in rows] + [
        float(row["stddev"]) for row in rows
    ]


def infer(features, checkpoint_path):
    """Inference only; no pretrained checkpoint or training results are supplied."""
    import torch
    from torch import nn

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hidden = int(checkpoint["hidden_dim"])
    classes = int(checkpoint["num_classes"])
    if hidden <= 0 or classes < 2:
        raise ValueError("invalid checkpoint architecture")
    network = nn.Sequential(
        nn.Linear(len(features), hidden), nn.ReLU(), nn.Linear(hidden, classes)
    )
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.eval()
    with torch.no_grad():
        probabilities = network(
            torch.tensor([features], dtype=torch.float32)
        ).softmax(dim=1)[0]
    return {
        "class_index": int(probabilities.argmax().item()),
        "probabilities": [float(p) for p in probabilities],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    content, channels, samples = read_window(args.csv)
    features = extract_features(args.extractor, content, channels)
    output = {
        "channels": channels, "samples": samples,
        "features": features, "prediction": None,
    }
    if args.checkpoint is not None:
        output["prediction"] = infer(features, args.checkpoint)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
