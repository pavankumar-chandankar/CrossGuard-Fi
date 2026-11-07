"""Versioned recording manifest and provenance-aware, traversal-safe file resolution."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from crossguard_fi.config import CONTRACT


@dataclass(frozen=True)
class Recording:
    recording_id: str
    relative_path: str
    subject_id: str
    session_id: str
    domain_id: str
    label: int
    sampling_hz: int = CONTRACT.sampling_hz
    schema_version: int = CONTRACT.schema_version

    @classmethod
    def from_mapping(cls, item: dict) -> "Recording":
        required = {"recording_id", "relative_path", "subject_id", "session_id",
                    "domain_id", "label"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Manifest missing keys: {sorted(missing)}")
        if set(item) - set(cls.__dataclass_fields__):
            raise ValueError(f"Unexpected manifest keys: {sorted(set(item) - set(cls.__dataclass_fields__))}")
        record = cls(**item)
        for key in ("recording_id", "subject_id", "session_id", "domain_id"):
            if not isinstance(getattr(record, key), str) or not getattr(record, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        if type(record.label) is not int or not 0 <= record.label < CONTRACT.classes:
            raise ValueError("Label outside 0..23")
        if record.sampling_hz != CONTRACT.sampling_hz or record.schema_version != 1:
            raise ValueError("Incorrect sample rate or manifest schema version")
        if (not isinstance(record.relative_path, str)
                or not record.relative_path.endswith(".npz")):
            raise ValueError("Only .npz recordings supported")
        p = Path(record.relative_path)
        if p.is_absolute() or ".." in p.parts or not p.parts:
            raise ValueError("Unsafe relative_path; must stay inside the dataset root")
        return record


def load_manifest(manifest: Path) -> list[Recording]:
    records: list[Recording] = []
    used_ids: set[str] = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("Each entry must be a JSON object")
            record = Recording.from_mapping(obj)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Manifest line {number}: {exc}") from exc
        if record.recording_id in used_ids:
            raise ValueError(f"Duplicate recording ID: {record.recording_id}")
        records.append(record)
        used_ids.add(record.recording_id)
    if not records:
        raise ValueError("Empty manifest")
    return records


def resolve_recording(root: Path, record: Recording) -> Path:
    base = root.resolve(strict=True)
    candidate = (base / record.relative_path).resolve(strict=True)
    if not candidate.is_relative_to(base):
        raise ValueError(f"Recording escapes dataset root: {record.relative_path}")
    return candidate


def manifest_digest(records: Iterable[Recording]) -> str:
    payload = [r.__dict__ for r in sorted(records, key=lambda r: r.recording_id)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_digests(root: Path, records: Iterable[Recording]) -> dict[str, str]:
    """SHA-256 per source recording; protects evaluation against in-place dataset mutation."""
    digests = {}
    for record in records:
        path = resolve_recording(root, record)
        hasher = sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(block)
        digests[record.recording_id] = hasher.hexdigest()
    return digests
