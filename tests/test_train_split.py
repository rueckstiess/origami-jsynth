"""Tests for the 80/10/10 train/val/test split helper in `origami_jsynth.train`.

The data stage writes train.jsonl (90% of the dataset) and test.jsonl (10%) to
disk. `split_train_eval` carves a further 10%-overall validation slice out of
train.jsonl, leaving an 80% fit set — matching the split reported in the paper.
"""

from __future__ import annotations

import pytest

from origami_jsynth.train import split_train_eval


@pytest.fixture
def records_90pct() -> list[dict]:
    """90 records — what train.jsonl would hold for a 100-record dataset."""
    return [{"id": i} for i in range(90)]


def test_split_sizes_match_80_10_10(records_90pct: list[dict]) -> None:
    fit, eval_ = split_train_eval(records_90pct, seed=42)
    # 90 / 9 == 10  →  80 fit + 10 eval out of 90 (== 80/10 of 100 overall)
    assert len(fit) == 80
    assert len(eval_) == 10
    assert len(fit) + len(eval_) == len(records_90pct)


def test_split_partitions_records(records_90pct: list[dict]) -> None:
    fit, eval_ = split_train_eval(records_90pct, seed=42)
    fit_ids = {r["id"] for r in fit}
    eval_ids = {r["id"] for r in eval_}
    assert fit_ids.isdisjoint(eval_ids)
    assert fit_ids | eval_ids == {r["id"] for r in records_90pct}


def test_split_is_deterministic(records_90pct: list[dict]) -> None:
    fit_a, eval_a = split_train_eval(records_90pct, seed=42)
    fit_b, eval_b = split_train_eval(records_90pct, seed=42)
    assert [r["id"] for r in fit_a] == [r["id"] for r in fit_b]
    assert [r["id"] for r in eval_a] == [r["id"] for r in eval_b]


def test_split_varies_with_seed(records_90pct: list[dict]) -> None:
    _, eval_a = split_train_eval(records_90pct, seed=42)
    _, eval_b = split_train_eval(records_90pct, seed=7)
    # Different seeds should produce different eval sets (with overwhelming
    # probability for n=90, eval_size=10).
    assert {r["id"] for r in eval_a} != {r["id"] for r in eval_b}


def test_split_does_not_mutate_input(records_90pct: list[dict]) -> None:
    original = [dict(r) for r in records_90pct]
    split_train_eval(records_90pct, seed=42)
    assert records_90pct == original


def test_split_shuffles_before_taking_slice(records_90pct: list[dict]) -> None:
    """The eval slice must not just be the first N records of an ordered list."""
    _, eval_ = split_train_eval(records_90pct, seed=42)
    eval_ids = [r["id"] for r in eval_]
    # If shuffling worked, the eval ids won't all be in the first 10 positions
    # of the input, with overwhelming probability.
    assert set(eval_ids) != set(range(10))


def test_split_handles_tiny_input() -> None:
    """`max(1, ...)` guarantees at least one eval record even for very small inputs."""
    fit, eval_ = split_train_eval([{"id": 0}, {"id": 1}], seed=42)
    assert len(eval_) >= 1
    assert len(fit) + len(eval_) == 2
