"""Build a JSON API request from a validated or generated NPZ recording."""
import argparse
import json
from pathlib import Path

import numpy as np
from crossguard_fi.config import CONTRACT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.recording, allow_pickle=False) as data:
        csi = data["csi"][:CONTRACT.window_samples]
        timestamps = data["timestamps_ns"][:CONTRACT.window_samples]
    if csi.shape != (CONTRACT.window_samples, CONTRACT.channels):
        raise ValueError("Input does not contain a 100 x 270 window")
    args.output.write_text(json.dumps({"samples": csi.tolist(),
                                       "timestamps_ns": timestamps.tolist(),
                                       "top_k": 3}), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
