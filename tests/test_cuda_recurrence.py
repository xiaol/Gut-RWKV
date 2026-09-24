import pytest
import torch

from rwkv_jev.backbone.rwkv7 import _wkv7_loop
from rwkv_jev.backbone.wkv7_kernel import available, wkv7_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_recurrence_and_initial_state_gradients_match_reference():
    if not available(64):
        pytest.skip("CUDA toolchain required")
    generator = torch.Generator(device="cuda").manual_seed(42)
    shape = (2, 32, 2, 64)
    tensors = [torch.randn(shape, generator=generator, device="cuda") * 0.05 for _ in range(6)]
    tensors[1].fill_(-2)
    for tensor in tensors:
        tensor[:, 23:] = 0
    tensors[1][:, 23:] = -1e4
    initial = torch.randn((2, 2, 64, 64), generator=generator, device="cuda") * 0.05
    fused_inputs = [tensor.to(torch.bfloat16).requires_grad_() for tensor in tensors]
    fused_state = initial.clone().requires_grad_()
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in fused_inputs]
    reference_state = initial.clone().requires_grad_()
    fused, final = wkv7_cuda(*fused_inputs, fused_state)
    reference, reference_final = _wkv7_loop(*reference_inputs, reference_state)
    torch.testing.assert_close(fused.float(), reference, rtol=0.005, atol=2e-4)
    torch.testing.assert_close(final, reference_final, rtol=1e-4, atol=2e-6)
    upstream = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    fused.backward(upstream)
    reference.backward(upstream.float())
    torch.testing.assert_close(fused_state.grad, reference_state.grad, rtol=2e-4, atol=2e-6)
    assert fused_state.grad.abs().sum() > 0
    for actual, expected in zip(fused_inputs, reference_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0.02, atol=2e-4)
