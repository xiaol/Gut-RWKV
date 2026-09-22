// WKV-7 recurrence with an explicit initial state and its gradient.
//
// Forward:  S_t = S_{t-1} * diag(w_t) + (S_{t-1} a_t) b_t^T + v_t k_t^T ;  y_t = S_t r_t
// with w_t = exp(-exp(w_in)) (the reference RWKV-7 parametrisation, w_in <= -0.5).
// Thread i of a block owns row i of the state (the v / output index); j runs over the k index.
//
// Adapted from BlinkDL's wkv7_cuda.cu (decay parametrisation, dw formula) and RWKV-PEFT's
// rwkv7_state kernels (initial state s0 in, ds0 out).  The state is checkpointed every
// _CHUNK_LEN_ steps in s_ (stored transposed) and reconstructed in the backward pass.

#include <cuda_bf16.h>
#include <assert.h>

using bf = __nv_bfloat16;
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); }

using i64 = long long int;
typedef bf * __restrict__ F_;

template<int N> __launch_bounds__(N, 2)
__global__ void forward_kernel(int T, int H, const float* __restrict__ s0_, F_ r_, F_ w_, F_ k_, F_ v_, F_ a_, F_ b_, bf* __restrict__ y_, float* s__, float* __restrict__ sa_) {
    const int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);
    s0_ += i64(bb*H+hh) * i64(N*N) + i64(i*N);
    float state[N];
#pragma unroll
    for (int j = 0; j < N; ++j) state[j] = s0_[j];
    __shared__ float r[N], w[N], k[N], a[N], b[N];

    for (int t = 0; t < T; ++t) {
        const i64 idx = (i64(bb)*T + t) * i64(H)*N + i64(hh)*N + i;
        __syncthreads();
        r[i] = to_float(r_[idx]);
        w[i] = __expf(-__expf(to_float(w_[idx])));
        k[i] = to_float(k_[idx]);
        a[i] = to_float(a_[idx]);
        b[i] = to_float(b_[idx]);
        __syncthreads();

        float sa = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) sa += state[j] * a[j];
        sa_[idx] = sa;

        const float vi = to_float(v_[idx]);
        float y = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * vi;
            y += s * r[j];
            state[j] = s;
        }
        y_[idx] = to_bf(y);

        if ((t+1) % _CHUNK_LEN_ == 0) {
            const int base = (t/_CHUNK_LEN_)*N*N + i;
#pragma unroll
            for (int j = 0; j < N; ++j) s_[base + j*N] = state[j];
        }
    }
}

template<int N>
__global__ void backward_kernel(int T, int H, F_ r_, F_ w_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, const float* __restrict__ s__, const float* __restrict__ sa_,
                                float* ds0_, bf* dr_, bf* dw_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    const int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    const float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);
    ds0_ += i64(bb*H+hh) * i64(N*N) + i64(i*N);

    float stateT[N] = {0}, dstate[N] = {0}, dstateT[N] = {0};
    __shared__ float r[N], w[N], k[N], v[N], a[N], b[N], dy[N], sa[N], dSb_shared[N];
    float ri, wi, wi_fac, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; --t) {
        const i64 idx = (i64(bb)*T + t) * i64(H)*N + i64(hh)*N + i;
        __syncthreads();
        r[i] = ri = to_float(r_[idx]);
        wi_fac = -__expf(to_float(w_[idx]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[idx]);
        v[i] = to_float(v_[idx]);
        a[i] = ai = to_float(a_[idx]);
        b[i] = bi = to_float(b_[idx]);
        dy[i] = dyi = to_float(dy_[idx]);
        sa[i] = sa_[idx];
        __syncthreads();

        if ((t+1) % _CHUNK_LEN_ == 0) {
            const int base = (t/_CHUNK_LEN_)*N*N + i*N;
#pragma unroll
            for (int j = 0; j < N; ++j) stateT[j] = s_[base + j];
        }

        float dr = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) dr += stateT[j] * dy[j];
        dr_[idx] = to_bf(dr);

        const float iwi = 1.0f / wi;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * r[j];
            dstateT[j] += ri * dy[j];
        }

        float dw = 0.0f, dk = 0.0f, dv = 0.0f, db = 0.0f, dSb = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * v[j];
            dv += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            db += dstateT[j] * sa[j];
        }
        dw_[idx] = to_bf(dw * wi * wi_fac);
        dk_[idx] = to_bf(dk);
        dv_[idx] = to_bf(dv);
        db_[idx] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) da += stateT[j] * dSb_shared[j];
        da_[idx] = to_bf(da);

#pragma unroll
        for (int j = 0; j < N; ++j) {
            dstate[j] = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
        }
    }
#pragma unroll
    for (int j = 0; j < N; ++j) ds0_[j] = dstate[j];
}

void cuda_forward(int B, int T, int H, float* s0, bf* r, bf* w, bf* k, bf* v, bf* a, bf* b, bf* y, float* s, float* sa, cudaStream_t stream) {
    assert(T % _CHUNK_LEN_ == 0);
    forward_kernel<_N_><<<dim3(H, B), dim3(_N_), 0, stream>>>(T, H, s0, r, w, k, v, a, b, y, s, sa);
}
void cuda_backward(int B, int T, int H, bf* r, bf* w, bf* k, bf* v, bf* a, bf* b, bf* dy, float* s, float* sa, float* ds0, bf* dr, bf* dw, bf* dk, bf* dv, bf* da, bf* db, cudaStream_t stream) {
    assert(T % _CHUNK_LEN_ == 0);
    backward_kernel<_N_><<<dim3(H, B), dim3(_N_), 0, stream>>>(T, H, r, w, k, v, a, b, dy, s, sa, ds0, dr, dw, dk, dv, da, db);
}
