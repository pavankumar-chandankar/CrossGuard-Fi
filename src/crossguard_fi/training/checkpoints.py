"""Strict, atomic model artifacts, schema checks and provenance metadata."""
from hashlib import sha256
import os
from pathlib import Path
import torch

from crossguard_fi.config import CONTRACT
from crossguard_fi.data.normalization import ChannelNormalizer
from crossguard_fi.models.networks import build_model

ARTIFACT_SCHEMA = 1


def save_checkpoint(destination: Path, model: torch.nn.Module, metadata: dict) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"artifact_schema": ARTIFACT_SCHEMA, "contract": CONTRACT.to_dict(),
               "model": model.state_dict(), "metadata": metadata}
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return sha256(destination.read_bytes()).hexdigest()


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    # `weights_only` prohibits arbitrary Python object deserialization.
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if (payload.get("artifact_schema") != ARTIFACT_SCHEMA
            or payload.get("contract") != CONTRACT.to_dict()):
        raise ValueError("Incompatible artifact or CSI contract")
    metadata = payload["metadata"]
    needed = {"model", "hidden_dim", "dropout", "normalizer", "manifest_digest",
              "subject_split", "recording_split", "recording_sha256", "seed"}
    if not isinstance(metadata, dict) or not needed <= metadata.keys():
        raise ValueError("Missing model metadata")
    ChannelNormalizer.from_dict(metadata["normalizer"])
    model = build_model(metadata["model"], int(metadata["hidden_dim"]),
                        float(metadata["dropout"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)
    return model, metadata
