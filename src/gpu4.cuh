#ifndef GPU4_H
#define GPU4_H

#include "gpu2.cuh"

class GPU4 : public GPU
{
public:
    GPU4() = default;
    ~GPU4();
    void removeSingleSeam(uchar3 *inPixels, int width, int height, int seam_order, uchar3 *outPixels, int blocksize, Debuger *logger);
};
__global__ void reduceBlksKernel3(int *in, int n, int *out);

void GPU4::removeSingleSeam(uchar3 *inPixels, int width, int height, int seam_order, uchar3 *outPixels, int blocksize, Debuger *logger)
{
    dim3 blockSize(blocksize, blocksize);
    // 0. Preparation
    /* Declare variables */
    uchar3 *src = inPixels;

    // Variables
    uchar3 *d_src;
    int *d_src_gray, *d_src_blur, *d_energies, *d_min;

    int *min_path = (int *)malloc(height * sizeof(int));
    int *min_row0 = (int *)malloc(width * sizeof(int));
    int *min_array = (int *)malloc(width * height * sizeof(int));

    size_t pixelsSize_3channels = width * height * sizeof(uchar3);
    size_t pixelsSize_1channels = width * height * sizeof(int);
    size_t smem_size2alloc;

    CHECK(cudaMalloc(&d_src, pixelsSize_3channels));
    CHECK(cudaMemcpy(d_src, src, pixelsSize_3channels, cudaMemcpyHostToDevice));

    // 1. Convert img to gray
    CHECK(cudaMalloc(&d_src_gray, pixelsSize_1channels));
    dim3 gridSize_gray((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    convertRgb2Gray_kernel<<<gridSize_gray, blockSize>>>(d_src, width, height, d_src_gray);

    // 2. Blur gray img
    CHECK(cudaMalloc(&d_src_blur, pixelsSize_1channels));
    dim3 gridSize_blur((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    smem_size2alloc = (blockSize.x + BLUR_KERNEL_SIZE - 1) * (blockSize.y + BLUR_KERNEL_SIZE - 1) * sizeof(int);

    blurImg_kernel_v2<<<gridSize_blur, blockSize, smem_size2alloc>>>(d_src_gray, width, height, d_src_blur);

    // 3. Calc Energies
    CHECK(cudaMalloc(&d_energies, pixelsSize_1channels));
    smem_size2alloc = (blockSize.x + SOBEL_KERNEL_SIZE - 1) * (blockSize.y + SOBEL_KERNEL_SIZE - 1) * sizeof(int);

    calcEnergyMap_kernel_v2<<<gridSize_blur, blockSize, smem_size2alloc>>>(d_src_blur, width, height, d_energies);
    if (seam_order == 1)
    {
        int *h_src_gray = (int *)malloc(pixelsSize_1channels);
        CHECK(cudaMemcpy(h_src_gray, d_energies, pixelsSize_1channels, cudaMemcpyDeviceToHost));
        logger->drawEnergyImage(h_src_gray, width, height);
    }

    // 4. Find min cost each row iteratively (dp - each rows parallel)
    CHECK(cudaMalloc(&d_min, width * height * sizeof(int)));
    dim3 gridSize_SeamMap((width - 1) / blockSize.x + 1);

    for (int y = height - 2; y >= 0; y--)
    {
        computeMinEnergyRow_kernel<<<gridSize_SeamMap, blockSize>>>(d_energies, width, height, y, d_min);
    }

    CHECK(cudaMemcpy(min_row0, d_energies, width * sizeof(int), cudaMemcpyDeviceToHost));
    // 4. Reduce: find min cost on top row
    dim3 reduce_gridSize((width - 1) / (blockSize.x * 2) + 1);

    int *d_out_ids;
    int *h_out = (int *)malloc(reduce_gridSize.x * sizeof(int));
    CHECK(cudaMalloc(&d_out_ids, reduce_gridSize.x * sizeof(int)));

    smem_size2alloc = blockSize.x * 2 * sizeof(int);
    reduceBlksKernel3<<<reduce_gridSize, blockSize, smem_size2alloc>>>(d_energies, width, d_out_ids);

    CHECK(cudaMemcpy(h_out, d_out_ids, reduce_gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));

    int min_val = min_row0[h_out[0]];
    int min_seam_idx = h_out[0];
    for (int i = 1; i < reduce_gridSize.x; i++)
    {
        if (min_row0[h_out[i]] < min_val)
        {
            min_seam_idx = h_out[i];
            min_val = min_row0[h_out[i]];
        }
    }

    if (min_seam_idx == -1)
    {
        printf("cannot find min seam at %d\n", seam_order);
    }

    // 5. Calculate min col idx on each row
    CHECK(cudaMemcpy(min_array, d_min, width * height * sizeof(int), cudaMemcpyDeviceToHost));
    logger->drawSeamImage(inPixels, width, height, seam_order, min_seam_idx, min_array);
    min_path[0] = min_seam_idx;

    for (int i = 1; i < height; ++i)
    {
        min_path[i] = min_array[(i - 1) * width + min_path[i - 1]];
    }
    // 6. Remove scene
    uchar3 *d_output;
    int *d_min_path;

    CHECK(cudaMalloc(&d_output, (width - 1) * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_min_path, height * sizeof(int)));
    CHECK(cudaMemcpy(d_min_path, min_path, height * sizeof(int), cudaMemcpyHostToDevice));

    removeSeam<<<gridSize_blur, blockSize>>>(d_src, width, height, d_min_path, d_output);
    CHECK(cudaMemcpy(outPixels, d_output, (width - 1) * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    // Clean memory
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_energies));
    CHECK(cudaFree(d_src_blur));
    CHECK(cudaFree(d_src_gray));

    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_min));
    CHECK(cudaFree(d_min_path));
    CHECK(cudaFree(d_out_ids));

    free(min_array);
    free(min_path);
    free(min_row0);
    free(h_out);
}

__global__ void reduceBlksKernel3(int *in, int n, int *out)
{
    extern __shared__ int smem[];
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    if (idx < n)
    {
        smem[threadIdx.x * 2] = idx;
    }
    if (idx + 1 < n)
    {
        smem[threadIdx.x * 2 + 1] = idx + 1;
    }
    __syncthreads();

    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
    {
        if (threadIdx.x % stride == 0)
        {
            if (idx + stride < n)
            {
                if (in[smem[threadIdx.x * 2 + stride]] < in[smem[threadIdx.x * 2]])
                {
                    smem[threadIdx.x * 2] = smem[threadIdx.x * 2 + stride];
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        out[blockIdx.x] = smem[0];
    }
}

#endif /* GPU4_H */
