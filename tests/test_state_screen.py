from pathlib import Path
from types import SimpleNamespace

from scripts.state_screen import ARMS, POLICY_ARMS, build_commands, idle_gpu, wait_for_gpu


def test_idle_gpu_requires_both_free_memory_and_no_compute_process(monkeypatch):
    def output(command, **kwargs):
        if "--query-gpu=uuid,memory.free" in command:
            return "GPU-busy, 35000\nGPU-small, 2000\nGPU-idle, 40000\n"
        return "GPU-busy\n"

    monkeypatch.setattr("scripts.state_screen.subprocess.check_output", output)
    assert idle_gpu() == "GPU-idle"
    assert idle_gpu(minimum_free_mib=41000) is None
    assert idle_gpu(preferred_uuid="GPU-small") is None
    assert idle_gpu(preferred_uuid="GPU-busy") is None
    assert idle_gpu(preferred_uuid="GPU-idle") == "GPU-idle"


def test_screen_keeps_budget_fixed_and_interventions_separate(tmp_path):
    args = SimpleNamespace(output=tmp_path, base=Path("base"), vocab=Path("vocab"),
                           train=Path("train"), development=Path("dev"), subset=Path("subset"),
                           steps=1536, checkpoint_every=384, seed=42)
    for name, intervention in ARMS.items():
        training, evaluation = build_commands(tmp_path, args, name)
        assert training[training.index("--max-steps") + 1] == "1536"
        assert training[training.index("--seed") + 1] == "42"
        assert training[training.index("--lr") + 1] == "1e-4"
        assert "--state-tuning" in training
        assert evaluation[evaluation.index("--data") + 1] == "dev"
        for flag in ["--state-lr", "--state-max-norm", "--sampling"]:
            assert (flag in training) == (flag in intervention)


def test_waiting_screen_can_be_cancelled_without_gpu_access(tmp_path, monkeypatch):
    (tmp_path / "STOP").touch()
    monkeypatch.setattr("scripts.state_screen.idle_gpu", lambda: (_ for _ in ()).throw(AssertionError("GPU queried")))
    assert wait_for_gpu(tmp_path, True, "control") is None
    assert '"phase": "cancelled"' in (tmp_path / "status.json").read_text()


def test_policy_screen_matches_every_setting_except_estimator(tmp_path):
    args = SimpleNamespace(output=tmp_path, base=Path("base"), vocab=Path("vocab"),
                           train=Path("train"), development=Path("dev"), subset=Path("subset"),
                           steps=1536, checkpoint_every=384, seed=42, suite="policy")
    normalized = []
    for name in POLICY_ARMS:
        training, evaluation = build_commands(tmp_path, args, name)
        assert training[training.index("--objective") + 1] == name
        assert training[training.index("--state-lr") + 1] == "1e-5"
        assert training[training.index("--policy-samples") + 1] == "32"
        assert evaluation[evaluation.index("--data") + 1] == "dev"
        normalized.append([argument.replace(name, "ESTIMATOR") for argument in training])
    assert normalized[0] == normalized[1]
