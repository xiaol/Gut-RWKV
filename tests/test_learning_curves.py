import json

import pytest

from scripts.summarize_learning_curves import read_progress, summarize


def test_read_progress_and_summary(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "noise\n"
        '{"step": 1, "loss": 2.0}\n'
        '{"step": 10, "loss": 0.5, "reward": 0.2}\n',
        encoding="utf-8",
    )
    summary = tmp_path / "train-summary.json"
    summary.write_text(json.dumps({"steps": 12, "final_training_loss": 0.1, "initial_state_norm": 1.2}), encoding="utf-8")
    points = read_progress(log)
    assert [point["step"] for point in points] == [1, 10]
    assert summarize(points, summary)["final_step"] == 12
    assert summarize(points, summary)["minimum_loss_step"] == 10


def test_read_progress_rejects_bad_steps_and_nonfinite_values(tmp_path):
    duplicate = tmp_path / "duplicate.log"
    duplicate.write_text('{"step": 2, "loss": 1}\n{"step": 2, "loss": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_progress(duplicate)
    nonfinite = tmp_path / "nonfinite.log"
    nonfinite.write_text('{"step": 1, "loss": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        read_progress(nonfinite)
