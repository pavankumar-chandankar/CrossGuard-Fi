#!/usr/bin/env python3
"""Backward-compatible CSI CSV/C++ feature command.

This adapter is retained for the project's original feature-extraction CLI.
Research training and HTTP inference now live in `crossguard_fi`.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--extractor", required=True, type=Path)
    args = p.parse_args()
    content = args.csv.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(content)))
    if not rows or not rows[0]:
        raise ValueError("Empty CSI window")
    channels = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != channels or any(not math.isfinite(float(x)) for x in row):
            raise ValueError(f"Invalid CSI row {i + 1}")
    result = subprocess.run([str(args.extractor.resolve()), str(channels)], input=content,
                            capture_output=True, text=True, check=True)
    stats = list(csv.DictReader(io.StringIO(result.stdout)))
    features = [float(r["mean"]) for r in stats] + [float(r["stddev"]) for r in stats]
    print(json.dumps({"samples": len(rows), "channels": channels,
                      "features": features, "prediction": None}, indent=2))


if __name__ == "__main__":
    main()
