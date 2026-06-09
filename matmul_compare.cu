#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---- Naive kernel ----
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ---- Tiled kernel ----
__global__ void matmul_tiles(int N, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    float sum = 0.0f;

    for (int t = 0; t < 32; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * blockDim.x + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * blockDim.y + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < 16; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}

// ---- CPU reference ----
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[row * N + k] * B[k * N + col];
            C[row * N + col] = s;
        }
}

// Helper: time a kernel with one warm-up run, return milliseconds of the timed run.
float time_kernel(void (*launch)(dim3, dim3, int, float*, float*, float*),
                  dim3 grid, dim3 block, int N,
                  float* dA, float* dB, float* dC) {
    // warm-up (not timed)
    launch(grid, block, N, dA, dB, dC);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch(grid, block, N, dA, dB, dC);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Tiny wrappers so both kernels share one launch signature.
void launch_naive(dim3 g, dim3 b, int N, float* A, float* B, float* C) {
    matmul_naive<<<g, b>>>(A, B, C, N);
}
void launch_tiles(dim3 g, dim3 b, int N, float* A, float* B, float* C) {
    matmul_tiles<<<g, b>>>(N, A, B, C);
}

int main() {
    const int N = 512;
    const size_t bytes = (size_t)N * N * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    float *hRef = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        hA[i] = (float)rand() / RAND_MAX;
        hB[i] = (float)rand() / RAND_MAX;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    matmul_cpu(hA, hB, hRef, N);  // reference answer

    // ---- Naive ----
    float naive_ms = time_kernel(launch_naive, grid, block, N, dA, dB, dC);
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    double naive_diff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double d = fabs((double)hC[i] - (double)hRef[i]);
        if (d > naive_diff) naive_diff = d;
    }

    // ---- Tiled ----
    float tiled_ms = time_kernel(launch_tiles, grid, block, N, dA, dB, dC);
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    double tiled_diff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double d = fabs((double)hC[i] - (double)hRef[i]);
        if (d > tiled_diff) tiled_diff = d;
    }

    printf("Matrix size: %d x %d\n\n", N, N);
    printf("NAIVE  : %.3f ms   (max diff %.2e)  %s\n",
           naive_ms, naive_diff, naive_diff < 1e-3 ? "CORRECT" : "WRONG");
    printf("TILED  : %.3f ms   (max diff %.2e)  %s\n",
           tiled_ms, tiled_diff, tiled_diff < 1e-3 ? "CORRECT" : "WRONG");
    printf("\nSpeedup (naive / tiled): %.2fx\n", naive_ms / tiled_ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}