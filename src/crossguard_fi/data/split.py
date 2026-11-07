"""Subject-disjoint experiment partitioning; zero frame/window overlap across sets."""
from dataclasses import dataclass
import random

from crossguard_fi.data.manifest import Recording


@dataclass(frozen=True)
class SubjectSplit:
    train: tuple[Recording, ...]
    val: tuple[Recording, ...]
    test: tuple[Recording, ...]

    def subjects(self) -> dict[str, list[str]]:
        return {name: sorted({r.subject_id for r in getattr(self, name)})
                for name in ("train", "val", "test")}


def assert_disjoint(split: SubjectSplit) -> None:
    subjects = {name: set(ids) for name, ids in split.subjects().items()}
    if (subjects["train"] & subjects["val"] or subjects["train"] & subjects["test"]
            or subjects["val"] & subjects["test"]):
        raise ValueError("Subject leakage across train/validation/test sets")
    ids = [r.recording_id for r in (*split.train, *split.val, *split.test)]
    if len(ids) != len(set(ids)):
        raise ValueError("Recording appears in multiple subsets")
    if not split.train or not split.val or not split.test:
        raise ValueError("All three partitions must contain recordings")


def subject_holdout(records: list[Recording], seed: int = 17,
                    validation_subjects: int | None = None,
                    test_subjects: int | None = None) -> SubjectSplit:
    subjects = sorted({r.subject_id for r in records})
    if len(subjects) < 3:
        raise ValueError("Need >=3 subjects for disjoint train, val, and test")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = test_subjects if test_subjects is not None else max(1, round(0.2 * n))
    n_val = validation_subjects if validation_subjects is not None else max(1, round(0.2 * n))
    if n_test < 1 or n_val < 1 or n_test + n_val >= n:
        raise ValueError("Holdout counts must leave at least one training subject")
    t = set(subjects[:n_test])
    v = set(subjects[n_test:n_test + n_val])
    tr = set(subjects[n_test + n_val:])
    split = SubjectSplit(
        train=tuple(r for r in records if r.subject_id in tr),
        val=tuple(r for r in records if r.subject_id in v),
        test=tuple(r for r in records if r.subject_id in t),
    )
    assert_disjoint(split)
    return split
