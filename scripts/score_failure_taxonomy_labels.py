#!/usr/bin/env python3
"""Compute Cohen's kappa for two failure-taxonomy label CSV files."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _read_labels(path: Path, label_column: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = str(row.get("sample_id") or "").strip()
            label = str(row.get(label_column) or "").strip()
            if sample_id and label:
                labels[sample_id] = label
    return labels


def _cohens_kappa(left: list[str], right: list[str]) -> tuple[float, float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("label lists must be the same nonzero length")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum((left_counts[label] / len(left)) * (right_counts[label] / len(right)) for label in labels)
    if expected == 1.0:
        return 1.0, observed, expected
    return (observed - expected) / (1.0 - expected), observed, expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater-a", required=True)
    parser.add_argument("--rater-b", required=True)
    parser.add_argument("--label-column", default="rater_label")
    args = parser.parse_args()

    labels_a = _read_labels(Path(args.rater_a), args.label_column)
    labels_b = _read_labels(Path(args.rater_b), args.label_column)
    sample_ids = sorted(set(labels_a) & set(labels_b))
    if not sample_ids:
        raise SystemExit("no overlapping labeled sample_id values")

    left = [labels_a[sample_id] for sample_id in sample_ids]
    right = [labels_b[sample_id] for sample_id in sample_ids]
    kappa, observed, expected = _cohens_kappa(left, right)

    print(f"samples={len(sample_ids)}")
    print(f"observed_agreement={observed:.6f}")
    print(f"expected_agreement={expected:.6f}")
    print(f"cohens_kappa={kappa:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
