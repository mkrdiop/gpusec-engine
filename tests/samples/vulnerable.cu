#include <cuda_runtime.h>
#define BLOCK 256

__global__ void scale(float *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = in[i] * 2.0f;
}

__global__ void divergent_sync(float *data, int n) {
  __shared__ float tile[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  if (tid < 128) {
    tile[tid] = data[i];
    __syncthreads();
  }
}

__global__ void early_exit(float *data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  __syncthreads();
  data[i] += 1.0f;
}

__global__ void race(float *out, const float *in, int n) {
  __shared__ float s[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  s[tid] = (i < n) ? in[i] : 0.0f;
  if (i < n) out[i] = s[(tid + 1) % BLOCK];
}

__global__ void shared_oob(float *out) {
  __shared__ float buf[BLOCK];
  buf[BLOCK] = 1.0f;
  out[0] = buf[0];
}

__global__ void spin(volatile int *flag) {
  while (true) { }
}

__global__ void tail_loop(float *data, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    data[i] += 1.0f;
    __syncthreads();
  }
}

void host(float *d, size_t bytes) {
  cudaMalloc(&d, bytes);
  CUDA_CHECK(cudaFree(d));
  cudaFree(d);
}
