import json

import pytest
import torch

from rwkv_jev.schema import Question
from scripts.audit_decisions import audit_model, data_audit, question_key, read_questions


def write_record(path, label, criteria):
    record = {"state": "A customer asks for a refund", "questions": {
        "route": {"type": "choice", "instructions": "Route the message", "criteria": criteria,
                  "label": label, "src": "routing"}},
              "_meta": {"group_id": "customer-1", "source": "routing"}}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_overlap_and_conflicting_labels_ignore_candidate_order(tmp_path):
    training_path = tmp_path / "train.jsonl"
    development_path = tmp_path / "development.jsonl"
    write_record(training_path, "billing", {"billing": "Payments", "technical": "Software"})
    write_record(development_path, "technical", {"technical": "Software", "billing": "Payments"})
    training, development = read_questions(training_path), read_questions(development_path)
    assert question_key(training[0]) == question_key(development[0])
    audit = data_audit(training, development)
    assert audit["development_questions_with_training_state"] == 1
    assert audit["development_questions_with_training_group"] == 1
    assert audit["development_questions_with_training_question"] == 1
    assert audit["conflicting_question_keys_across_splits"] == 1


def test_reordered_targets_do_not_create_false_label_conflicts(tmp_path):
    training_path = tmp_path / "train.jsonl"
    development_path = tmp_path / "development.jsonl"
    write_record(training_path, "billing", {"billing": "Payments", "technical": "Software"})
    write_record(development_path, "billing", {"technical": "Software", "billing": "Payments"})
    audit = data_audit(read_questions(training_path), read_questions(development_path))
    assert audit["conflicting_question_keys_across_splits"] == 0


def test_metrics_grouping_subset_and_cache_disagreement():
    question = Question.parse({"type": "noul", "instructions": "Is it true?"})
    rows = [{"id": "1:label", "instance_key": "first", "state": "first", "source": "one", "question": question, "target": 1},
            {"id": "2:label", "instance_key": "second", "state": "second", "source": "two", "question": question, "target": 0}]

    class FixedModel:
        temperature = 1.0

        def logits(self, state, question, use_cache=False):
            return torch.tensor([0.25, 0.75] if use_cache else [0.75, 0.25]).log()

    result, outcomes = audit_model(FixedModel(), rows, {rows[0]["instance_key"]}, 1)
    assert result["accuracy"] == 0.5
    assert result["brier"] == pytest.approx(0.625)
    assert result["by_source"]["one"]["accuracy"] == 1
    assert result["by_source"]["two"]["accuracy"] == 0
    assert result["by_type"]["noul"]["questions"] == 2
    assert result["pilot_subset"]["questions"] == 1
    assert result["pilot_subset"]["accuracy"] == 1
    assert result["cache_parity"]["argmax_disagreements"] == 2
    assert result["cache_parity"]["max_probability_difference"] == pytest.approx(0.5)
    assert [row["pilot_subset"] for row in outcomes] == [True, False]
    with pytest.raises(ValueError, match="absent"):
        audit_model(FixedModel(), rows, {"missing"}, 0)
    duplicated = [rows[0], {**rows[0], "id": "3:label"}]
    result, outcomes = audit_model(FixedModel(), duplicated, {rows[0]["instance_key"]}, 0)
    assert result["pilot_subset"]["questions"] == 1
    assert [row["pilot_subset"] for row in outcomes] == [True, False]
