"""Tests for TabDiff checkpoint selection logic (_find_checkpoint)."""

import pytest
from tabdiff.train import _find_checkpoint


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class TestFindCheckpoint:
    def test_prefers_lower_loss_across_ema_and_regular(self, tmp_path):
        # EMA has higher loss than regular — regular should win.
        _touch(tmp_path / "best_ema_model_2.5000_500.pt")
        _touch(tmp_path / "best_model_1.2000_400.pt")
        result = _find_checkpoint(str(tmp_path))
        assert result.endswith("best_model_1.2000_400.pt")

    def test_prefers_ema_when_it_has_lower_loss(self, tmp_path):
        _touch(tmp_path / "best_ema_model_0.8000_800.pt")
        _touch(tmp_path / "best_model_1.5000_600.pt")
        result = _find_checkpoint(str(tmp_path))
        assert result.endswith("best_ema_model_0.8000_800.pt")

    def test_multiple_best_files_picks_lowest_loss(self, tmp_path):
        _touch(tmp_path / "best_ema_model_3.0000_100.pt")
        _touch(tmp_path / "best_ema_model_1.5000_200.pt")
        _touch(tmp_path / "best_model_2.0000_300.pt")
        _touch(tmp_path / "best_model_0.9000_400.pt")
        result = _find_checkpoint(str(tmp_path))
        assert result.endswith("best_model_0.9000_400.pt")

    def test_falls_back_to_final_ema_when_no_best(self, tmp_path):
        _touch(tmp_path / "final_ema_model.pt")
        _touch(tmp_path / "final_model.pt")
        result = _find_checkpoint(str(tmp_path))
        assert result.endswith("final_ema_model.pt")

    def test_falls_back_to_final_model_when_no_best_or_final_ema(self, tmp_path):
        _touch(tmp_path / "final_model.pt")
        result = _find_checkpoint(str(tmp_path))
        assert result.endswith("final_model.pt")

    def test_best_takes_priority_over_final(self, tmp_path):
        _touch(tmp_path / "best_model_1.0000_100.pt")
        _touch(tmp_path / "final_ema_model.pt")
        _touch(tmp_path / "final_model.pt")
        result = _find_checkpoint(str(tmp_path))
        assert result.endswith("best_model_1.0000_100.pt")

    def test_raises_when_no_checkpoints(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No TabDiff checkpoints"):
            _find_checkpoint(str(tmp_path))

    def test_raises_when_directory_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No TabDiff checkpoints"):
            _find_checkpoint(str(tmp_path / "nonexistent"))
