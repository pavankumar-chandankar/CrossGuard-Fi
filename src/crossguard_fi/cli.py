"""CrossGuard-Fi command line: dataset, training, evaluation and local inference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from crossguard_fi.config import TrainConfig, CONTRACT
from crossguard_fi.data.manifest import load_manifest
from crossguard_fi.data.synthetic import generate_fixture
from crossguard_fi.data.validation import load_recording, validate_dataset
from crossguard_fi.serving.predictor import Predictor
from crossguard_fi.training.engine import evaluate, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crossguard", description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("generate-fixture", help="Synthetic data ONLY for CI/smoke tests")
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--subjects", type=int, default=4)
    synth.add_argument("--trials-per-class", type=int, default=1)
    synth.add_argument("--samples", type=int, default=110)
    synth.add_argument("--seed", type=int, default=17)

    def add_dataset_args(target):
        target.add_argument("--manifest", type=Path, required=True)
        target.add_argument("--data-root", type=Path, required=True)

    validate = commands.add_parser("validate", help="Audit a complete manifest and recording set")
    add_dataset_args(validate)

    trainer = commands.add_parser("train", help="Train with train/validation only; seal test")
    add_dataset_args(trainer)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--model", choices=["tcn", "mlp"], default="tcn")
    trainer.add_argument("--seed", type=int, default=17)
    trainer.add_argument("--epochs", type=int, default=12)
    trainer.add_argument("--batch-size", type=int, default=32)
    trainer.add_argument("--learning-rate", type=float, default=0.001)
    trainer.add_argument("--patience", type=int, default=4)
    trainer.add_argument("--hidden-dim", type=int, default=64)
    trainer.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")

    loso = commands.add_parser("loso", help="Subject-wise outer test folds with independent validation")
    add_dataset_args(loso)
    loso.add_argument("--output", type=Path, required=True)
    loso.add_argument("--model", choices=["tcn", "mlp"], default="mlp")
    loso.add_argument("--seed", type=int, default=17)
    loso.add_argument("--epochs", type=int, default=12)
    loso.add_argument("--batch-size", type=int, default=32)
    loso.add_argument("--hidden-dim", type=int, default=64)
    loso.add_argument("--max-folds", type=int, default=None,
                      help="Optional partial smoke run; omitted means all subjects")

    test = commands.add_parser("evaluate", help="Evaluate saved, subject-disjoint sealed test set")
    add_dataset_args(test)
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")

    infer = commands.add_parser("predict", help="Predict from first 100 samples of an NPZ recording")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--recording", type=Path, required=True)
    infer.add_argument("--top-k", type=int, default=3)

    benchmark = commands.add_parser("benchmark", help="Local inference latency only, NOT accuracy")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--runs", type=int, default=30)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate-fixture":
        path = generate_fixture(args.output, args.subjects, args.trials_per_class,
                                args.samples, args.seed)
        report = {"manifest": str(path), "synthetic_only": True,
                  "note": "Do not report synthetic accuracy in a CV or paper"}
    elif args.command == "validate":
        report = validate_dataset(args.data_root, load_manifest(args.manifest))
    elif args.command == "train":
        cfg = TrainConfig(seed=args.seed, model=args.model, epochs=args.epochs,
                          batch_size=args.batch_size, learning_rate=args.learning_rate,
                          patience=args.patience, hidden_dim=args.hidden_dim,
                          device=args.device)
        report = train(args.manifest, args.data_root, args.output, cfg)
    elif args.command == "loso":
        from crossguard_fi.training.loso import run_loso
        cfg = TrainConfig(seed=args.seed, model=args.model, epochs=args.epochs,
                          batch_size=args.batch_size, hidden_dim=args.hidden_dim)
        report = run_loso(args.manifest, args.data_root, args.output, cfg, args.max_folds)
    elif args.command == "evaluate":
        report = evaluate(args.checkpoint, args.manifest, args.data_root,
                          args.output, device=args.device)
    elif args.command == "predict":
        with np.load(args.recording, allow_pickle=False) as values:
            sample = values["csi"][:CONTRACT.window_samples]
            timestamps = values["timestamps_ns"][:CONTRACT.window_samples]
        report = Predictor(args.checkpoint).predict(sample, timestamps, args.top_k)
    elif args.command == "benchmark":
        if args.runs < 1:
            raise ValueError("runs must be positive")
        predictor = Predictor(args.checkpoint)
        zeros = np.zeros((CONTRACT.window_samples, CONTRACT.channels), dtype=np.float32)
        timestamps = np.arange(CONTRACT.window_samples, dtype=np.int64) * 20_000_000
        for _ in range(3):
            predictor.predict(zeros, timestamps)
        timings = []
        for _ in range(args.runs):
            started = time.perf_counter()
            predictor.predict(zeros, timestamps)
            timings.append((time.perf_counter() - started) * 1000)
        report = {"latency_p50_ms": float(np.percentile(timings, 50)),
                  "latency_p95_ms": float(np.percentile(timings, 95)),
                  "runs": args.runs, "device": "cpu", "synthetic_input": True,
                  "note": "Local benchmark only. Hardware-specific; not real deployment performance."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
