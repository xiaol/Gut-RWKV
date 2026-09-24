from types import SimpleNamespace

import pytest
import torch

from rwkv_jev import DecisionModel, Question
from rwkv_jev.backbone.rwkv7 import Rwkv7ForCausalLM
from rwkv_jev.cli import gaussian_policy_loss, objective_loss, proper_score_reward, train


class ByteTokenizer:
    def encode(self, text):
        return [value + 1 for value in text.encode()]


QUESTION = {"type": "choice", "instructions": "Pick a color", "criteria": {"red": "R", "blue": "B"}}


def make_model():
    torch.manual_seed(12)
    config = SimpleNamespace(hidden_size=16, num_hidden_layers=2, vocab_size=257,
                             head_size=8, num_heads=2, dim_ffn=32,
                             d_decay=4, d_aaa=4, d_mv=4, d_gate=4)
    backbone = Rwkv7ForCausalLM(config)
    for parameter in backbone.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    backbone.grad_checkpoint = False
    return DecisionModel(backbone, ByteTokenizer(), head_size=16, state_tuning=True)


def test_proper_score_rewards_correct_distribution_and_ordinal_order():
    correct = proper_score_reward(torch.tensor([8.0, -2.0, -2.0]), 0)
    uniform = proper_score_reward(torch.zeros(3), 0)
    wrong = proper_score_reward(torch.tensor([-2.0, -2.0, 8.0]), 0)
    assert correct > uniform > wrong
    low = proper_score_reward(torch.tensor([0.0, 4.0, -4.0]), 1)
    high = proper_score_reward(torch.tensor([0.0, -4.0, 4.0]), 1)
    assert low > high
    loss, reward = objective_loss(torch.tensor([0.2, -0.1], requires_grad=True), 0, "proper-score")
    loss.backward()
    assert reward is not None and torch.isfinite(loss)


def test_rlcd_objective_is_seeded_and_baseline_centered():
    torch.manual_seed(99)
    logits = torch.tensor([0.2, -0.1], requires_grad=True)
    first, reward = objective_loss(logits, 0, "rlcd", exploration_std=0.1)
    torch.manual_seed(99)
    repeat, repeated_reward = objective_loss(logits.detach().clone().requires_grad_(), 0, "rlcd", exploration_std=0.1)
    torch.testing.assert_close(first, repeat)
    torch.testing.assert_close(reward, repeated_reward)
    centered, _ = objective_loss(logits, 0, "rlcd", exploration_std=0.1, baseline=reward)
    centered.backward()
    assert torch.isfinite(centered)
    with pytest.raises(ValueError, match="exploration"):
        objective_loss(logits, 0, "rlcd", exploration_std=0)
    with pytest.raises(ValueError, match="Unknown"):
        objective_loss(logits, 0, "unknown")


def test_training_rlcd_updates_state_and_checkpoint_callback():
    model = make_model()
    initial = model.initial_wkv.detach().clone()
    progress = []
    records = [("red", Question.parse(QUESTION), 0)]
    train(model, records, epochs=1, learning_rate=0.01, seed=12, objective="rlcd", exploration_std=0.1,
          checkpoint_every=1, checkpoint_callback=lambda current, epoch: progress.append(current.trained_steps))
    assert progress == [1]
    assert model.trained_steps == 1
    assert not torch.equal(initial, model.initial_wkv)


def test_reparameterized_baseline_changes_loss_but_not_gradient():
    logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    baseline = torch.tensor(1.25, requires_grad=True)
    torch.manual_seed(99)
    loss, reward = objective_loss(logits, 0, "rlcd", exploration_std=0.1)
    gradient = torch.autograd.grad(loss, logits)[0]
    torch.manual_seed(99)
    centered, repeated_reward = objective_loss(logits, 0, "rlcd", exploration_std=0.1, baseline=baseline)
    centered_gradient, baseline_gradient = torch.autograd.grad(centered, (logits, baseline), allow_unused=True)
    torch.testing.assert_close(reward, repeated_reward)
    torch.testing.assert_close(centered.detach(), loss.detach() + baseline.detach())
    torch.testing.assert_close(gradient, centered_gradient)
    assert baseline_gradient is None


def test_training_objective_validation():
    records = [("red", Question.parse(QUESTION), 0)]
    with pytest.raises(ValueError, match="Unknown"):
        train(make_model(), records, epochs=1, learning_rate=0.01, seed=12, objective="other")
    with pytest.raises(ValueError, match="decay"):
        train(make_model(), records, epochs=1, learning_rate=0.01, seed=12, objective="rlcd", baseline_decay=1)


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_reinforce_matches_expected_pathwise_gradient(kind):
    logits = torch.tensor([0.4, -0.3, 0.1], requires_grad=True)
    gradients, rewards = [], []
    for estimator in ("pathwise", "reinforce"):
        torch.manual_seed(37)
        loss, reward = gaussian_policy_loss(logits, 1, kind, estimator, policy_samples=65536)
        gradients.append(torch.autograd.grad(loss, logits)[0])
        rewards.append(reward)
    torch.testing.assert_close(rewards[0], rewards[1])
    torch.testing.assert_close(gradients[0], gradients[1], atol=0.012, rtol=0.025)
    assert gradients[1][1] < 0


def test_unordered_reward_is_permutation_invariant_and_stable():
    logits = torch.tensor([[0.3, -0.7, 1.2], [1.1, 0.1, -0.4]], requires_grad=True)
    permutation = torch.tensor([2, 0, 1])
    original = proper_score_reward(logits, 1, ordered=False, stable_log=True)
    reordered = proper_score_reward(logits[:, permutation], 2, ordered=False, stable_log=True)
    torch.testing.assert_close(original, reordered)
    gradient = torch.autograd.grad(original.sum(), logits)[0]
    reordered_gradient = torch.autograd.grad(reordered.sum(), logits)[0]
    torch.testing.assert_close(gradient, reordered_gradient)
    extreme = torch.tensor([1000.0, -1000.0], requires_grad=True)
    reward = proper_score_reward(extreme, 1, ordered=False, stable_log=True)
    assert torch.isfinite(reward)
    torch.testing.assert_close(torch.autograd.grad(reward, extreme)[0], torch.tensor([-1.0, 1.0]))


@pytest.mark.parametrize("estimator", ["reinforce", "pathwise"])
def test_policy_training_updates_state_and_uses_typed_reward(estimator):
    model = make_model()
    initial = model.initial_wkv.detach().clone()
    records = [("red", Question.parse(QUESTION), 0)]
    train(model, records, epochs=1, learning_rate=0.01, seed=12, objective=estimator)
    assert model.trained_steps == 1
    assert not torch.equal(initial, model.initial_wkv)
    logits = torch.tensor([0.2, -0.3, 0.5], requires_grad=True)
    for kind in ("choice", "noul"):
        torch.manual_seed(13)
        without_rps = gaussian_policy_loss(logits, 0, kind, estimator, rps_weight=0)[0]
        torch.manual_seed(13)
        with_rps = gaussian_policy_loss(logits, 0, kind, estimator, rps_weight=100)[0]
        torch.testing.assert_close(without_rps, with_rps)
    with pytest.raises(ValueError, match="even"):
        gaussian_policy_loss(logits, 0, "choice", estimator, policy_samples=3)
    with pytest.raises(ValueError, match="question kind"):
        gaussian_policy_loss(logits, 0, None, estimator)
