"""Tests for evaluation metrics (fidelity, detection, privacy, utility)."""

import random

import pytest

from origami_jsynth.evaluation.detection import DetectionResult, compute_detection
from origami_jsynth.evaluation.evaluate import EvaluationResult
from origami_jsynth.evaluation.fidelity import FidelityResult, compute_fidelity
from origami_jsynth.evaluation.privacy import PrivacyResult, compute_privacy
from origami_jsynth.evaluation.utility import UtilityResult, compute_utility

# ---------------------------------------------------------------------------
# Helper data generators
# ---------------------------------------------------------------------------


def make_records(n=100, seed=42):
    """Generate small flat records for testing."""
    rng = random.Random(seed)
    return [
        {
            "age": rng.randint(20, 70),
            "income": rng.gauss(50000, 10000),
            "label": rng.choice(["A", "B", "C"]),
        }
        for _ in range(n)
    ]


def make_nested_records(n=100, seed=42):
    """Generate nested records with variable-length tag arrays."""
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        n_tags = rng.randint(0, 3)
        records.append(
            {
                "title": rng.choice(["foo", "bar", "baz"]),
                "score": rng.gauss(0, 1),
                "tags": [rng.choice(["x", "y", "z"]) for _ in range(n_tags)],
            }
        )
    return records


# ---------------------------------------------------------------------------
# TestComputeFidelity
# ---------------------------------------------------------------------------


class TestComputeFidelity:
    def test_identical_data_score_close_to_1(self):
        records = make_records(n=100)
        result = compute_fidelity(records, records, max_workers=1)
        assert result.overall_score >= 0.9, (
            f"Identical data should have high fidelity, got {result.overall_score:.4f}"
        )

    def test_different_distributions_score_lower(self):
        real = make_records(n=100, seed=1)
        # Synth from a clearly different distribution (different seed, different ranges).
        rng = random.Random(99)
        synth = [
            {"age": rng.randint(50, 90), "income": rng.gauss(100000, 5000), "label": rng.choice(["X", "Y"])}
            for _ in range(100)
        ]
        result = compute_fidelity(real, synth, max_workers=1)
        assert result.overall_score < 0.95

    def test_returns_fidelity_result(self):
        records = make_records(n=50)
        result = compute_fidelity(records, records, max_workers=1)
        assert isinstance(result, FidelityResult)

    def test_result_has_required_fields(self):
        records = make_records(n=50)
        result = compute_fidelity(records, records, max_workers=1)
        assert hasattr(result, "overall_score")
        assert hasattr(result, "column_shapes_score")
        assert hasattr(result, "column_pair_trends_score")

    def test_to_dict_from_dict_roundtrip(self):
        records = make_records(n=50)
        result = compute_fidelity(records, records, max_workers=1)
        d = result.to_dict()
        restored = FidelityResult.from_dict(d)
        assert restored.overall_score == pytest.approx(result.overall_score)
        assert restored.column_shapes_score == pytest.approx(result.column_shapes_score)
        assert restored.column_pair_trends_score == pytest.approx(result.column_pair_trends_score)

    def test_nested_records_work(self):
        records = make_nested_records(n=50)
        result = compute_fidelity(records, records, max_workers=1)
        assert isinstance(result, FidelityResult)
        assert result.overall_score >= 0.0

    def test_fidelity_result_to_dict_from_dict_no_compute(self):
        """Roundtrip to_dict/from_dict without running the full pipeline."""
        result = FidelityResult(
            overall_score=0.85,
            column_shapes_score=0.88,
            column_pair_trends_score=0.82,
            field_scores={"age": 0.9, "income": 0.8},
            field_details={"age": {"score": 0.9}, "income": {"score": 0.8}},
            field_weights={"age": 0.5, "income": 0.5},
            pair_scores={"age|income": 0.7},
            pair_details={"age|income": {"metric": "ks"}},
            pair_weights={"age|income": 1.0},
        )
        d = result.to_dict()
        restored = FidelityResult.from_dict(d)
        assert restored.overall_score == pytest.approx(0.85)
        assert restored.column_shapes_score == pytest.approx(0.88)
        assert restored.column_pair_trends_score == pytest.approx(0.82)
        assert restored.field_scores == result.field_scores
        assert restored.field_details == result.field_details


# ---------------------------------------------------------------------------
# TestComputeDetection
# ---------------------------------------------------------------------------


class TestComputeDetection:
    def test_identical_data_less_detectable_than_different(self):
        records = make_records(n=200)
        rng = random.Random(99)
        very_different = [
            {"age": rng.randint(60, 90), "income": rng.gauss(200000, 5000), "label": rng.choice(["X", "Y"])}
            for _ in range(200)
        ]
        identical_result = compute_detection(records, records)
        different_result = compute_detection(records, very_different)
        # Identical data should be harder to detect than very different data
        assert identical_result.detection_score >= different_result.detection_score

    def test_very_different_data_roc_auc_high(self):
        real = make_records(n=100, seed=1)
        rng = random.Random(99)
        synth = [
            {
                "age": rng.randint(60, 90),
                "income": rng.gauss(200000, 5000),
                "label": rng.choice(["X", "Y"]),
            }
            for _ in range(100)
        ]
        result = compute_detection(real, synth)
        assert result.roc_auc > 0.5

    def test_returns_detection_result(self):
        records = make_records(n=50)
        result = compute_detection(records, records)
        assert isinstance(result, DetectionResult)

    def test_result_has_required_fields(self):
        records = make_records(n=50)
        result = compute_detection(records, records)
        assert hasattr(result, "detection_score")
        assert hasattr(result, "roc_auc")
        assert hasattr(result, "feature_importances")

    def test_to_dict_from_dict_roundtrip(self):
        records = make_records(n=50)
        result = compute_detection(records, records)
        d = result.to_dict()
        restored = DetectionResult.from_dict(d)
        assert restored.detection_score == pytest.approx(result.detection_score)
        assert restored.roc_auc == pytest.approx(result.roc_auc)
        assert restored.feature_importances == result.feature_importances

    def test_detection_result_to_dict_from_dict_no_compute(self):
        result = DetectionResult(
            detection_score=0.75,
            roc_auc=0.62,
            method="xgboost",
            feature_importances={"age.num": 0.5, "income.num": 0.3},
        )
        d = result.to_dict()
        restored = DetectionResult.from_dict(d)
        assert restored.detection_score == pytest.approx(0.75)
        assert restored.roc_auc == pytest.approx(0.62)
        assert restored.method == "xgboost"
        assert restored.feature_importances == result.feature_importances


# ---------------------------------------------------------------------------
# TestComputePrivacy
# ---------------------------------------------------------------------------


class TestComputePrivacy:
    def test_returns_privacy_result(self):
        records = make_records(n=50)
        train = records[:30]
        test = records[30:40]
        synth = make_records(n=10, seed=99)
        result = compute_privacy(train, test, synth, verbose=False)
        assert isinstance(result, PrivacyResult)

    def test_result_has_required_fields(self):
        records = make_records(n=60)
        train, test = records[:30], records[30:40]
        synth = make_records(n=10, seed=99)
        result = compute_privacy(train, test, synth, verbose=False)
        assert hasattr(result, "privacy_score")
        assert hasattr(result, "dcr_score")
        assert hasattr(result, "exact_matches_train")
        assert hasattr(result, "exact_matches_test")

    def test_exact_duplicates_detected(self):
        train = make_records(n=20, seed=1)
        test = make_records(n=10, seed=2)
        # Synth is an exact copy of training records.
        synth = train[:5]
        result = compute_privacy(train, test, synth, verbose=False)
        assert result.exact_matches_train >= 5

    def test_to_dict_from_dict_roundtrip(self):
        records = make_records(n=60)
        train, test = records[:30], records[30:40]
        synth = make_records(n=10, seed=99)
        result = compute_privacy(train, test, synth, verbose=False)
        d = result.to_dict()
        restored = PrivacyResult.from_dict(d)
        assert restored.privacy_score == pytest.approx(result.privacy_score)
        assert restored.dcr_score == pytest.approx(result.dcr_score)
        assert restored.exact_matches_train == result.exact_matches_train
        assert restored.exact_matches_test == result.exact_matches_test

    def test_privacy_result_to_dict_from_dict_no_compute(self):
        result = PrivacyResult(
            privacy_score=0.9,
            dcr_score=55.0,
            exact_matches_train=2,
            exact_matches_test=0,
            exact_matches_train_only=2,
            ref_size=100,
        )
        d = result.to_dict()
        restored = PrivacyResult.from_dict(d)
        assert restored.privacy_score == pytest.approx(0.9)
        assert restored.dcr_score == pytest.approx(55.0)
        assert restored.exact_matches_train == 2
        assert restored.exact_matches_test == 0
        assert restored.exact_matches_train_only == 2
        assert restored.ref_size == 100


# ---------------------------------------------------------------------------
# TestComputeUtility
# ---------------------------------------------------------------------------


class TestComputeUtility:
    def test_returns_utility_result(self):
        train = make_records(n=80, seed=1)
        test = make_records(n=20, seed=2)
        synth = make_records(n=80, seed=3)
        result = compute_utility(train, test, synth, target_field="label", task_type="classification")
        assert isinstance(result, UtilityResult)

    def test_utility_score_in_range(self):
        train = make_records(n=80, seed=1)
        test = make_records(n=20, seed=2)
        synth = make_records(n=80, seed=3)
        result = compute_utility(train, test, synth, target_field="label", task_type="classification")
        assert 0.0 <= result.utility_score <= 1.0

    def test_result_has_required_fields(self):
        train = make_records(n=80, seed=1)
        test = make_records(n=20, seed=2)
        synth = make_records(n=80, seed=3)
        result = compute_utility(train, test, synth, target_field="label", task_type="classification")
        assert hasattr(result, "trtr_metrics")
        assert hasattr(result, "tstr_metrics")
        assert hasattr(result, "utility_score")

    def test_to_dict_from_dict_roundtrip(self):
        train = make_records(n=80, seed=1)
        test = make_records(n=20, seed=2)
        synth = make_records(n=80, seed=3)
        result = compute_utility(train, test, synth, target_field="label", task_type="classification")
        d = result.to_dict()
        restored = UtilityResult.from_dict(d)
        assert restored.utility_score == pytest.approx(result.utility_score)
        assert restored.trtr_metrics == result.trtr_metrics
        assert restored.tstr_metrics == result.tstr_metrics

    def test_utility_result_to_dict_from_dict_no_compute(self):
        result = UtilityResult(
            task_type="classification",
            target_column="label",
            trtr_metrics={"accuracy": 0.8, "f1_weighted": 0.78},
            tstr_metrics={"accuracy": 0.75, "f1_weighted": 0.72},
            utility_score=0.92,
        )
        d = result.to_dict()
        restored = UtilityResult.from_dict(d)
        assert restored.utility_score == pytest.approx(0.92)
        assert restored.task_type == "classification"
        assert restored.target_column == "label"
        assert restored.trtr_metrics == result.trtr_metrics
        assert restored.tstr_metrics == result.tstr_metrics


# ---------------------------------------------------------------------------
# TestEvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def _make_full_result(self):
        """Build a complete EvaluationResult directly without running any pipeline."""
        fidelity = FidelityResult(
            overall_score=0.85,
            column_shapes_score=0.88,
            column_pair_trends_score=0.82,
            field_scores={"age": 0.9},
            field_details={"age": {"score": 0.9}},
        )
        detection = DetectionResult(
            detection_score=0.7,
            roc_auc=0.65,
            method="xgboost",
            feature_importances={"age.num": 0.4},
        )
        privacy = PrivacyResult(
            privacy_score=0.95,
            dcr_score=52.0,
            exact_matches_train=0,
            exact_matches_test=0,
            exact_matches_train_only=0,
            ref_size=100,
        )
        utility = UtilityResult(
            task_type="classification",
            target_column="label",
            trtr_metrics={"f1_weighted": 0.8},
            tstr_metrics={"f1_weighted": 0.75},
            utility_score=0.9375,
        )
        metrics = {
            "fidelity": 0.85,
            "detection": 0.7,
            "privacy": 0.95,
            "utility": 0.9375,
        }
        details = {
            "fidelity": fidelity.to_dict(),
            "detection": detection.to_dict(),
            "privacy": privacy.to_dict(),
            "utility": utility.to_dict(),
        }
        return EvaluationResult(
            metrics=metrics,
            details=details,
            fidelity_result=fidelity,
            detection_result=detection,
            privacy_result=privacy,
            utility_result=utility,
        )

    def test_from_dict_to_dict_roundtrip_preserves_metrics(self):
        result = self._make_full_result()
        d = result.to_dict()
        restored = EvaluationResult.from_dict(d)
        assert restored.metrics == result.metrics

    def test_from_dict_to_dict_roundtrip_preserves_sub_results(self):
        result = self._make_full_result()
        d = result.to_dict()
        restored = EvaluationResult.from_dict(d)
        assert restored.fidelity_result is not None
        assert restored.fidelity_result.overall_score == pytest.approx(0.85)
        assert restored.detection_result is not None
        assert restored.detection_result.detection_score == pytest.approx(0.7)
        assert restored.privacy_result is not None
        assert restored.privacy_result.privacy_score == pytest.approx(0.95)
        assert restored.utility_result is not None
        assert restored.utility_result.utility_score == pytest.approx(0.9375)

    def test_repr_includes_metric_names_and_values(self):
        result = self._make_full_result()
        r = repr(result)
        assert "fidelity" in r
        assert "detection" in r
        assert "privacy" in r
        assert "utility" in r
        # Numeric values should appear formatted.
        assert "0.85" in r or "0.8500" in r
