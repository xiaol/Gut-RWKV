from types import SimpleNamespace

import pytest
import torch
from rwkv_jev.backbone.rwkv7 import Rwkv7ForCausalLM

from rwkv_jev import DecisionModel, Question
from rwkv_jev.cli import evaluate, train


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
