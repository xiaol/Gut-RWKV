#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

using bf = __nv_bfloat16;

void cuda_forward(int B, int T, int H, float* s0, bf* r, bf* w, bf* k, bf* v, bf* a, bf* b, bf* y, float* s, float* sa, cudaStream_t stream);
void cuda_backward(int B, int T, int H, bf* r, bf* w, bf* k, bf* v, bf* a, bf* b, bf* dy, float* s, float* sa, float* ds0, bf* dr, bf* dw, bf* dk, bf* dv, bf* da, bf* db, cudaStream_t stream);

static void check(const torch::Tensor& t, const char* name, c10::ScalarType dt, const torch::Device& dev) {
    TORCH_CHECK(t.is_cuda() && t.device() == dev, name, " must be on ", dev);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == dt, name, " has wrong dtype");
}

void forward(torch::Tensor &s0, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    const auto dev = r.device();
    const c10::cuda::OptionalCUDAGuard guard(dev);
    for (auto* t : {&r, &w, &k, &v, &a, &b, &y}) check(*t, "bf16 input", torch::kBFloat16, dev);
    for (auto* t : {&s0, &s, &sa}) check(*t, "fp32 buffer", torch::kFloat32, dev);
    const int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    auto stream = c10::cuda::getCurrentCUDAStream(dev.index()).stream();
    cuda_forward(B, T, H, (float*)s0.data_ptr(), (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(), stream);
}

void backward(torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &dy, torch::Tensor &s, torch::Tensor &sa,
              torch::Tensor &ds0, torch::Tensor &dr, torch::Tensor &dw, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db) {
    const auto dev = r.device();
    const c10::cuda::OptionalCUDAGuard guard(dev);
    for (auto* t : {&r, &w, &k, &v, &a, &b, &dy, &dr, &dw, &dk, &dv, &da, &db}) check(*t, "bf16 tensor", torch::kBFloat16, dev);
    for (auto* t : {&s, &sa, &ds0}) check(*t, "fp32 buffer", torch::kFloat32, dev);
    const int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    auto stream = c10::cuda::getCurrentCUDAStream(dev.index()).stream();
    cuda_backward(B, T, H, (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(),
                  (float*)ds0.data_ptr(), (bf*)dr.data_ptr(), (bf*)dw.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr(), stream);
}

TORCH_LIBRARY(wkv7_state_N_N_, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
