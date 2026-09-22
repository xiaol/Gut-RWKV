"""JIT-compiled CUDA WKV-7 recurrence with initial-state gradients.

``wkv7_cuda(r, w, k, v, a, b, s0)`` mirrors ``rwkv7.wkv7`` (inputs [B,T,H,N], w pre-exponent,
state [B,H,N,N]; returns y [B,T,H,N] and the final state) but runs the time loop in one CUDA
kernel per layer (~100x faster than the Python loop) and back-propagates into ``s0``, which
is what a state bridge trains through.  Requires ``T % 16 == 0`` and head size ``N`` fixed at
compile time (one build per N, cached under ``~/.cache/state_bridge``).  Falls back
gracefully: ``available()`` is False when nvcc/ninja are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

CHUNK_LEN = 16
_OPS: dict[int, object] = {}
_FAILED: set[int] = set()
_SRC = Path(__file__).resolve().parent / "cuda"


def _load(N: int):
    if N in _OPS:
        return _OPS[N]
    if N in _FAILED:
        return None
    try:
        from torch.utils.cpp_extension import load

        build_dir = Path(os.environ.get("STATE_BRIDGE_KERNEL_DIR", Path.home() / ".cache" / "state_bridge")) / f"wkv7_state_N{N}"
        build_dir.mkdir(parents=True, exist_ok=True)
        cpp = (_SRC / "wkv7_state.cpp").read_text().replace("wkv7_state_N_N_", f"wkv7_state_N{N}")
        cpp_path = build_dir / "wkv7_state.cpp"
        if not cpp_path.exists() or cpp_path.read_text() != cpp:
            cpp_path.write_text(cpp)
        load(
            name=f"wkv7_state_N{N}",
            sources=[str(cpp_path), str(_SRC / "wkv7_state.cu")],
            is_python_module=False,
            verbose=False,
            build_directory=str(build_dir),
            extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={N}", f"-D_CHUNK_LEN_={CHUNK_LEN}"],
        )
        _OPS[N] = getattr(torch.ops, f"wkv7_state_N{N}")
        return _OPS[N]
    except Exception as e:  # pragma: no cover - depends on the build toolchain
        import traceback
        import warnings

        detail = traceback.format_exc() if os.environ.get("STATE_BRIDGE_KERNEL_DEBUG") else f"{type(e).__name__}: {str(e)[:300]}"
        warnings.warn(f"WKV-7 CUDA kernel unavailable for N={N} ({detail}); using the PyTorch loop")
        _FAILED.add(N)
        return None


def available(N: int) -> bool:
    return torch.cuda.is_available() and _load(N) is not None


class _WKV7State(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s0, r, w, k, v, a, b):
        B, T, H, N = r.shape
        op = _load(N)
        r, w, k, v, a, b = (t.to(torch.bfloat16).contiguous() for t in (r, w, k, v, a, b))
        s0 = s0.float().contiguous()
        y = torch.empty(B, T, H, N, dtype=torch.bfloat16, device=r.device)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=r.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)
        op.forward(s0, r, w, k, v, a, b, y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        final = s[:, :, -1].transpose(-1, -2).contiguous()  # stored transposed by the kernel
        ctx.mark_non_differentiable(final)
        return y, final

    @staticmethod
    def backward(ctx, dy, _dfinal):
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        B, T, H, N = r.shape
        op = _load(N)
        dy = dy.to(torch.bfloat16).contiguous()
        ds0 = torch.empty(B, H, N, N, dtype=torch.float32, device=r.device)
        grads = [torch.empty_like(r) for _ in range(6)]
        op.backward(r, w, k, v, a, b, dy, s, sa, ds0, *grads)
        return (ds0, *grads)


def wkv7_cuda(r, w, k, v, a, b, s0):
    """y, final_state = wkv7_cuda(...).  Gradients flow to r,w,k,v,a,b and s0 (not through final_state)."""
    if r.shape[1] % CHUNK_LEN != 0:
        raise ValueError(f"WKV-7 kernel needs T % {CHUNK_LEN} == 0, got T={r.shape[1]}")
    return _WKV7State.apply(s0, r, w, k, v, a, b)
