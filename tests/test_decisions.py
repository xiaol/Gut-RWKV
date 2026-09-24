from types import SimpleNamespace
from collections import Counter
import random

import pytest
import torch
from rwkv_jev.backbone.rwkv7 import Rwkv7ForCausalLM

from rwkv_jev import DecisionModel, Question
from rwkv_jev.cli import evaluate, objective_loss, proper_score_reward, train, training_order


class ByteTokenizer:
    def encode(self, text):
        return [value + 1 for value in text.encode()]


def make_model(rank=0, state_tuning=False):
    torch.manual_seed(12)
    config = SimpleNamespace(hidden_size=16, num_hidden_layers=2, vocab_size=257,
                             head_size=8, num_heads=2, dim_ffn=32,
                             d_decay=4, d_aaa=4, d_mv=4, d_gate=4)
    backbone = Rwkv7ForCausalLM(config)
    for parameter in backbone.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    backbone.grad_checkpoint = False
    return DecisionModel(backbone, ByteTokenizer(), rank=rank, head_size=16, candidate_batch=2,
                         state_tuning=state_tuning)


CHOICE = {"type": "choice", "instructions": "Pick a color", "criteria": {"red": "R", "blue": "B", "green": "G"}}


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_shared_state_matches_complete_sequences_and_isolation():
    model = make_model()
    question = Question.parse(CHOICE)
    with torch.no_grad():
        complete = model.logits("red", question)
        cached = model.logits("red", question, use_cache=True)
    torch.testing.assert_close(cached, complete, atol=2e-6, rtol=2e-5)
    model.trained_steps = 1
    alone = model.system_one("red", {"color": CHOICE})["answers"]["color"]
    together = model.system_one("red", {"other": {"type": "noul", "instructions": "Blue?"}, "color": CHOICE})["answers"]["color"]
    assert alone["probabilities"] == pytest.approx(together["probabilities"], abs=2e-6)


def test_choice_permutation_and_prefix_names():
    model = make_model()
    question = {**CHOICE, "criteria": {"cat": "animal", "cat bed": "furniture", "cat beds": "plural"}}
    reverse = {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
    with torch.no_grad():
        original = model.logits("pet", Question.parse(question)).softmax(-1)
        permuted = model.logits("pet", Question.parse(reverse)).softmax(-1).flip(0)
    torch.testing.assert_close(original, permuted)


def test_lora_training_freezes_base_and_adapter_roundtrip(tmp_path):
    model = make_model(rank=2)
    baseline = model.backbone.blocks[0].att.key.base.weight.detach().clone()
    records = [("red", Question.parse(CHOICE), 0)]
    train(model, records, epochs=2, learning_rate=0.01, seed=12)
    torch.testing.assert_close(model.backbone.blocks[0].att.key.base.weight, baseline)
    assert model.backbone.blocks[-1].ffn.value.up.weight.grad.abs().sum() > 0
    assert model.head[1].weight.grad.abs().sum() > 0
    path = tmp_path / "adapter.pt"
    model.save_adapter(path, {"test": True})
    restored = make_model(rank=2)
    assert restored.load_adapter(path) == {"test": True}
    assert restored.system_one("red", {"q": CHOICE}) == model.system_one("red", {"q": CHOICE})
    assert evaluate(model, records)["questions"] == 1


def test_adaptation_modes_start_with_identical_head_and_predictions():
    models = [make_model(), make_model(state_tuning=True), make_model(rank=2)]
    with torch.no_grad():
        for model in models[1:]:
            for name, weight in models[0].head.state_dict().items():
                torch.testing.assert_close(model.head.state_dict()[name], weight, rtol=0, atol=0)
            torch.testing.assert_close(model.logits("red", Question.parse(CHOICE)),
                                       models[0].logits("red", Question.parse(CHOICE)), rtol=0, atol=0)


def test_checkpoint_schedule_preserves_training():
    records = [("red", Question.parse(CHOICE), 0)] * 3
    baseline = make_model(state_tuning=True)
    observed = []
    checkpointed = make_model(state_tuning=True)
    train(baseline, records, epochs=2, learning_rate=0.01, seed=12, max_steps=5)
    train(checkpointed, records, epochs=2, learning_rate=0.01, seed=12, max_steps=5,
          checkpoint_every=2, checkpoint_callback=lambda model, epoch: observed.append((model.trained_steps, epoch)))
    assert observed == [(2, 1), (3, 1), (4, 2), (5, 2)]
    for name, weight in baseline.state_dict().items():
        torch.testing.assert_close(checkpointed.state_dict()[name], weight, rtol=0, atol=0)


def test_state_learning_rate_leaves_first_head_update_unchanged():
    baseline = make_model(state_tuning=True)
    constrained = make_model(state_tuning=True)
    records = [("red", Question.parse(CHOICE), 0)]
    train(baseline, records, epochs=1, learning_rate=0.01, seed=12)
    train(constrained, records, epochs=1, learning_rate=0.01, seed=12, state_learning_rate=0.001)
    for name, weight in baseline.head.state_dict().items():
        torch.testing.assert_close(constrained.head.state_dict()[name], weight, rtol=0, atol=0)
    torch.testing.assert_close(constrained.initial_wkv, baseline.initial_wkv * 0.1)


def test_state_norm_projection_and_invalid_controls():
    model = make_model(state_tuning=True)
    observed = []
    records = [("red", Question.parse(CHOICE), 0)]
    train(model, records, epochs=3, learning_rate=0.01, seed=12, state_max_norm=0.001,
          checkpoint_every=1, checkpoint_callback=lambda current, epoch: observed.append(float(current.initial_wkv.detach().norm())))
    assert all(0 < norm <= 0.001001 for norm in observed)
    with pytest.raises(ValueError, match="State controls"):
        train(make_model(), records, epochs=1, learning_rate=0.01, seed=12, state_learning_rate=0.001)
    with pytest.raises(ValueError, match="State controls"):
        train(model, records, epochs=1, learning_rate=0.01, seed=12, state_max_norm=float("nan"))


def test_balanced_sampling_preserves_sources_and_nonboolean_positions():
    noul = Question.parse({"type": "noul", "instructions": "True?"})
    choice = Question.parse(CHOICE)
    records = [("text", noul, target) for target in [0] * 9 + [1] + [0] + [1] * 9]
    records += [("text", choice, 0)] * 4
    sources = ["first"] * 10 + ["second"] * 10 + ["choice"] * 4
    natural = training_order(records, random.Random(42))
    balanced = training_order(records, random.Random(42), "noul-balanced", sources)
    assert balanced == training_order(records, random.Random(42), "noul-balanced", sources)
    assert [sources[index] for index in balanced] == [sources[index] for index in natural]
    for original, sampled in zip(natural, balanced):
        if records[original][1].kind != "noul":
            assert sampled == original
    for length in range(1, len(balanced) + 1):
        for source in ["first", "second"]:
            counts = Counter(records[index][2] for index in balanced[:length] if sources[index] == source)
            assert abs(counts[0] - counts[1]) <= 1
    with pytest.raises(ValueError, match="both labels"):
        training_order(records[:9], random.Random(42), "noul-balanced", sources[:9])
    with pytest.raises(ValueError, match="source"):
        training_order(records, random.Random(42), "noul-balanced")


def test_repeated_sample_index_does_not_trigger_early_epoch_checkpoint(monkeypatch):
    monkeypatch.setattr("rwkv_jev.cli.training_order", lambda *args: [0, 0, 1, 0])
    observed = []
    train(make_model(state_tuning=True), [("red", Question.parse(CHOICE), 0)] * 4,
          epochs=1, learning_rate=0.01, seed=12, checkpoint_every=3,
          checkpoint_callback=lambda model, epoch: observed.append(model.trained_steps))
    assert observed == [3, 4]


def test_validation_and_untrained_guard():
    model = make_model()
    with pytest.raises(RuntimeError, match="untrained"):
        model.system_one("red", {"q": CHOICE})
    model.settings["max_tokens"] = 10
    with pytest.raises(ValueError, match="max_tokens"):
        model.logits("red", Question.parse(CHOICE))
    with pytest.raises(ValueError, match="boolean"):
        Question.parse({"type": "noul", "instructions": "true?"}).target("true")
    with pytest.raises(ValueError, match="level"):
        Question.parse({"type": "score", "instructions": "rate", "criteria": ["low", "high"]}).target(True)


def test_typed_answers():
    score = Question.parse({"type": "score", "instructions": "rate", "criteria": ["low", "medium", "high"]})
    assert score.answer([0.2, 0.3, 0.5])["score"] == 1.3
    noul = Question.parse({"type": "noul", "instructions": "true?", "criteria": {"true": "yes"}})
    assert noul.answer([0.2, 0.8]) == {"type": "noul", "noul": 0.8}
    single = Question.parse({**CHOICE, "criteria": {"only": None}})
    assert single.answer([1.0])["confidence"] == 1.0
    with pytest.raises(ValueError, match="sum"):
        score.answer([0.2, 0.2, 0.2])


def test_state_tuning_gradients_and_roundtrip(tmp_path):
    model = make_model(state_tuning=True)
    initial = model.initial_wkv.detach().clone()
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    train(model, [("red", Question.parse(CHOICE), 0)], epochs=2, learning_rate=0.01, seed=12)
    assert model.initial_wkv.grad is not None
    assert model.initial_wkv.grad.abs().sum() > 0
    assert not torch.equal(initial, model.initial_wkv)
    path = tmp_path / "state.pt"
    model.save_adapter(path)
    restored = make_model(state_tuning=True)
    restored.load_adapter(path)
    assert restored.system_one("red", {"q": CHOICE}) == model.system_one("red", {"q": CHOICE})
    with torch.no_grad():
        question = Question.parse(CHOICE)
        torch.testing.assert_close(model.logits("red", question), model.logits("red", question, use_cache=True),
                                   atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="state tuning or LoRA"):
        make_model(rank=2, state_tuning=True)


def test_batched_questions_match_serial_and_reordered():
    model = make_model(state_tuning=True)
    model.trained_steps = 1
    questions = {"choice": CHOICE, "noul": {"type": "noul", "instructions": "Red?"},
                 "score": {"type": "score", "instructions": "Rate", "criteria": ["bad", "good"]}}
    model.settings["candidate_batch"] = 3
    together = model.system_one("red", questions)["answers"]
    reverse = model.system_one("red", dict(reversed(list(questions.items()))))["answers"]
    for name, definition in questions.items():
        alone = model.system_one("red", {name: definition})["answers"][name]
        field = "noul" if name == "noul" else "probabilities"
        assert together[name][field] == pytest.approx(alone[field], abs=2e-6)
        assert together[name][field] == pytest.approx(reverse[name][field], abs=2e-6)
