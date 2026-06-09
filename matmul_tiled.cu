#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void attention_naive(int N, int d,
                                 const float* Q, const float* K,
                                 const float* V, float* Out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // which token (row)
    //dot proding one row gives us the attention of one word to every other word
    float scores[128];
    float sum = 0;
    float maxScore = -INFINITY; 
    if (i >= N) return;
    
    // Step 1: scores[j] = dot(Q row i, K row j), for all j
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d+k] * K[j*d+k];
        
        if (dot > maxScore) {
            maxScore = dot; 
        }
        scores[j] = dot
    }

    // Step 2: softmax over scores[j]
    for (int p = 0; p < N; p++) {
        scores[p] = expf(scores[p]-maxScore);
        sum += scores[p];
    }

    for (int p = 0; p < N; p++) {
        scores[p] /= sum;
    }

    //weight is N long matrix, row is 
    // Step 3: Out row i = sum_j (weight[j] * V row j)
    //now output row which highlights value of word (d long matrix) with highest weight
    for (int c = 0; c < d; c++) {
        float out = 0;
        for (int k = 0; k < N; k++) {
            out += weight[k] * V[k*d+c]
        }
        Out[i][c] = out;

    }
    
}

__global__ void matmul_tiles(int N, const float* A, const float* B, float* C) {
    //this is the row and col of the output C 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //setup the tile that will shift just to calculate that output cell
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    //init sum to 0
    float sum = 0.0f;

    
    for (int t = 0; t < 32; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * blockDim.x + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * blockDim.y + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[row * N + k] * B[k * N + col];
            C[row * N + col] = s;
        }
}

int main() {
    const int N = 512;
    const size_t bytes = (size_t)N * N * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC_gpu = (float*)malloc(bytes);
    float *hC_cpu = (float*)malloc(bytes);

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
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_tiles<<<grid, block>>>(N, dA, dB, dC);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC_gpu, dC, bytes, cudaMemcpyDeviceToHost);
    matmul_cpu(hA, hB, hC_cpu, N);

    double max_diff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double d = fabs((double)hC_gpu[i] - (double)hC_cpu[i]);
        if (d > max_diff) max_diff = d;
    }

    printf("N = %d\n", N);
    printf("GPU kernel time: %.3f ms\n", ms);
    printf("Max difference vs CPU reference: %.6e\n", max_diff);
    printf(max_diff < 1e-3 ? "RESULT: CORRECT\n" : "RESULT: WRONG\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC_gpu); free(hC_cpu);
    return 0;
}