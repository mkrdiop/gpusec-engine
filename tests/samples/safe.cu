#define TILE 16

__global__ void scale_ok(float *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = in[i] * 2.0f;
}

__global__ void grid_stride(float *out, const float *in, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = in[i];
}

__global__ void reduce(const float *in, float *out, int n) {
  extern __shared__ float sdata[];
  unsigned tid = threadIdx.x;
  unsigned i = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void uniform_branch(float *data, int n, int mode) {
  __shared__ float t[64];
  if (mode > 0) { __syncthreads(); }
}

__global__ void matmul(const float *A, const float *B, float *C, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
    As[threadIdx.y][threadIdx.x] = (row < N && t * TILE + threadIdx.x < N) ? A[row * N + t * TILE + threadIdx.x] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (col < N && t * TILE + threadIdx.y < N) ? B[(t * TILE + threadIdx.y) * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < N && col < N) C[row * N + col] = acc;
}

__global__ void suppressed(float *out) {
  int i = threadIdx.x;
  out[i] = 1.0f;  // gpusec-ignore GS-MEM-001 host launches exactly one block of size n
}

void host(float *d, size_t bytes) {
  CUDA_CHECK(cudaMalloc(&d, bytes));
  (void)cudaFree(d);
}
