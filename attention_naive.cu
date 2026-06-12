#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void attention_flash(int N, int d,
                                 const float* Q, const float* K,
                                 const float* V, float* Out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // which token (row)
    if (i >= N) return;

    const int TILE = 16;   // process K/V in chunks of 16 tokens at a time

    // Running softmax state for THIS token (start values):
    float m = -INFINITY;   // running max score
    float l = 0.0f;        // running sum of exp(score - m)
    float O[64];           // running output (d long, assumes d <= 64)  
    for (int c = 0; c < d; c++) O[c] = 0.0f;

    // Slide through K/V in tiles of TILE tokens
    for (int t = 0; t < N; t += TILE) {

        // --- For each token j in this tile ---
        for (int j = t; j < t + TILE && j < N; j++) {

            // 1. Compute score = dot(Q row i, K row j) / sqrt(d)
            //    (you've done this before)
            float dot = 0.0f
            for (int k = 0; k < d; k++) {
                dot += (Q[i*d+k] * K[j*d+k]) / sqrt(d);
            }
            scores[j] = dot;

            // 2. Find new running max:  newM = max(m, score)

            // 3. Rescale factor for OLD state: correction = exp(m - newM)
            //    - rescale l:  l *= correction
            //    - rescale every O[c]: O[c] *= correction

            // 4. This token's weight: p = exp(score - newM)
            //    - add to sum:   l += p
            //    - add to output: O[c] += p * V[j*d + c]   for each c

            // 5. Update running max:  m = newM
        }
    }

    // After all tiles: normalize and write out
    for (int c = 0; c < d; c++)
        Out[i*d + c] = O[c] / l;
}

__global__ void attention_naive(int N, int d,
                                 const float* Q, const float* K,
                                 const float* V, float* Out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // which token (row)
    if (i >= N) return;

    float scores[128];                                // assumes N <= 128
    float maxScore = -INFINITY;
    float sum = 0.0f;

    // Step 1: scores[j] = dot(Q row i, K row j)
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        scores[j] = dot / sqrtf((float)d);
        if (dot > maxScore) maxScore = dot;
    }

    // Step 2: stable softmax
    for (int j = 0; j < N; j++) {
        scores[j] = expf(scores[j] - maxScore);
        sum += scores[j];
    }
    for (int j = 0; j < N; j++)
        scores[j] /= sum;

    // Step 3: Out row i = sum_j (scores[j] * V row j)
    for (int c = 0; c < d; c++) {
        float out = 0.0f;
        for (int j = 0; j < N; j++)
            out += scores[j] * V[j*d + c];
        Out[i*d + c] = out;
    }
}

// ---- CPU reference ----
void attention_cpu(int N, int d, const float* Q, const float* K,
                   const float* V, float* Out) {
    float* scores = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        float maxScore = -INFINITY, sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            scores[j] = dot/sqrt((float)d);
            if (dot > maxScore) maxScore = dot;
        }
        for (int j = 0; j < N; j++) {
            scores[j] = expf(scores[j] - maxScore);
            sum += scores[j];
        }
        for (int j = 0; j < N; j++) scores[j] /= sum;
        for (int c = 0; c < d; c++) {
            float out = 0.0f;
            for (int j = 0; j < N; j++)
                out += scores[j] * V[j*d + c];
            Out[i*d + c] = out;
        }
    }
    free(scores);
}

int main() {
    const int N = 128;       // sequence length (<=128 for the local scores array)
    const int d = 64;        // head dimension
    const size_t bytes = (size_t)N * d * sizeof(float);

    float *hQ = (float*)malloc(bytes);
    float *hK = (float*)malloc(bytes);
    float *hV = (float*)malloc(bytes);
    float *hOut_gpu = (float*)malloc(bytes);
    float *hOut_cpu = (float*)malloc(bytes);

    for (int i = 0; i < N * d; ++i) {
        hQ[i] = (float)rand() / RAND_MAX - 0.5f;
        hK[i] = (float)rand() / RAND_MAX - 0.5f;
        hV[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    float *dQ, *dK, *dV, *dOut;
    cudaMalloc(&dQ, bytes); cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes); cudaMalloc(&dOut, bytes);
    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (N + threads - 1) / threads;

    // warm-up
    attention_naive<<<blocks, threads>>>(N, d, dQ, dK, dV, dOut);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    attention_naive<<<blocks, threads>>>(N, d, dQ, dK, dV, dOut);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hOut_gpu, dOut, bytes, cudaMemcpyDeviceToHost);
    attention_cpu(N, d, hQ, hK, hV, hOut_cpu);

    double max_diff = 0.0;
    for (int i = 0; i < N * d; ++i) {
        double diff = fabs((double)hOut_gpu[i] - (double)hOut_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("N = %d, d = %d\n", N, d);
    printf("GPU kernel time: %.4f ms\n", ms);
    printf("Max difference vs CPU reference: %.6e\n", max_diff);
    printf(max_diff < 1e-3 ? "RESULT: CORRECT\n" : "RESULT: WRONG\n");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
    free(hQ); free(hK); free(hV); free(hOut_gpu); free(hOut_cpu);
    return 0;
}