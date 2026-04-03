#include "ImageProcessor.h"
#include <vector>

__global__ void sumImages(unsigned char *inputImages, unsigned int *imageSums,
                          int width, int height) {

  __shared__ unsigned int blockPixels[16][16]; // shared memory for block sum

  int localX = threadIdx.x; // thread X inside block
  int localY = threadIdx.y; // thread Y inside block

  int pixelX = blockIdx.x * blockDim.x + localX; // global pixel X
  int pixelY = blockIdx.y * blockDim.y + localY; // global pixel Y
  int imageId = blockIdx.z;                      // index of the image

  if (pixelX >= width || pixelY >= height)
    return;

  int imageSize = width * height;
  int pixelIndex = imageId * imageSize + pixelY * width + pixelX;
  blockPixels[localY][localX] = inputImages[pixelIndex];

  __syncthreads();

  // Horizontal reduction: sum each row in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (localX < stride) {
      blockPixels[localY][localX] += blockPixels[localY][localX + stride];
    }
    __syncthreads();
  }

  // Vertical reduction: sum across rows
  if (localX == 0) {
    for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
      if (localY < stride) {
        blockPixels[localY][0] += blockPixels[localY + stride][0];
      }
      __syncthreads();
    }

    // Final sum for this block
    if (localY == 0) {
      atomicAdd(&imageSums[imageId], blockPixels[0][0]);
    }
  }
}

// CUDA kernel: sum pixels in areas for each image
__global__ void areaSums(unsigned char *inputImages, unsigned int *areaSums,
                         int width, int height, int divider) {

  __shared__ unsigned int blockPixels[16][16];

  int localX = threadIdx.x;
  int localY = threadIdx.y;

  int pixelX = blockIdx.x * blockDim.x + localX;
  int pixelY = blockIdx.y * blockDim.y + localY;
  int imageId = blockIdx.z;

  if (pixelX < width && pixelY < height) {
    int imageSize = width * height;
    int pixelIndex = imageId * imageSize + pixelY * width + pixelX;
    blockPixels[localY][localX] = inputImages[pixelIndex];

    __syncthreads();

    int areaWidth = width / divider;
    int areaHeight = height / divider;
    int areaX = pixelX / areaWidth;
    int areaY = pixelY / areaHeight;

    for (int stride = 1; stride < min(blockDim.x, areaWidth); stride *= 2) {
      if (localX % (stride * 2) == 0) {
        blockPixels[localY][localX] += blockPixels[localY][localX + stride];
      }
      __syncthreads();
    }

    if (localX % areaWidth == 0) {
      for (int stride = 1; stride < min(blockDim.y, areaHeight); stride *= 2) {
        if (localY % (stride * 2) == 0) {
          blockPixels[localY][localX] += blockPixels[localY + stride][localX];
        }
        __syncthreads();
      }

      if (localY % areaHeight == 0) {
        atomicAdd(
            &areaSums[divider * divider * imageId + areaY * divider + areaX],
            blockPixels[localY][localX]);
      }
    }
  }
}

// CUDA kernel: divide accumulated sums by number of pixels
__global__ void divideAreas(unsigned int *areaSums, unsigned char *areaMeans,
                            int numAreas, int pixelsPerArea) {

  int imageId = blockIdx.y;
  int areaIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (areaIndex >= numAreas)
    return;

  int index = imageId * numAreas + areaIndex;
  areaMeans[index] = areaSums[index] / pixelsPerArea;
}

// Compute local means for image areas
std::vector<unsigned char> ImageProcessor::getLocalMeans(const std::vector<unsigned char>& imageData, int width, int height, int divider) {

    size_t imageSize = width * height;
    size_t pixelsPerArea = (width / divider) * (height / divider);
    size_t numAreas = divider * divider;
    int numImages = imageData.size() / imageSize;
    int blockSize = 16;

    dim3 blockDims(blockSize, blockSize);
    dim3 gridDims((width + blockSize - 1) / blockSize,
                    (height + blockSize - 1) / blockSize, numImages);

    int threadsPerBlock = 256;
    dim3 divBlock(threadsPerBlock);
    dim3 divGrid((numAreas + threadsPerBlock - 1) / threadsPerBlock, numImages);

    unsigned char *d_input;
    cudaMalloc((void **)&d_input, numImages * imageSize * sizeof(unsigned char));

    unsigned int *d_areaSums;
    cudaMalloc((void **)&d_areaSums, numImages * numAreas * sizeof(unsigned int));

    unsigned char *d_areaMeans;
    cudaMalloc((void **)&d_areaMeans, numImages * numAreas * sizeof(unsigned char));
    cudaMemcpy(d_input, imageData.data(), numImages * imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    areaSums<<<gridDims, blockDims>>>(d_input, d_areaSums, width, height, divider);
    cudaDeviceSynchronize();
    divideAreas<<<divGrid, divBlock>>>(d_areaSums, d_areaMeans, numAreas, pixelsPerArea);
    cudaDeviceSynchronize();
    std::vector<unsigned char> hostMeans(numImages * numAreas);
    cudaMemcpy(hostMeans.data(), d_areaMeans, numImages * numAreas * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_areaSums);
    cudaFree(d_areaMeans);

    return hostMeans;
}

// Compute mean intensity for entire images
std::vector<unsigned char> ImageProcessor::getGlobalMeans(const std::vector<unsigned char>& imagesData, int width, int height) {

  size_t numImages = imagesData.size() / (width * height);
  size_t imageSize = width * height;
  int blockSize = 16;
  dim3 blockDims(blockSize, blockSize);
  dim3 gridDims((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize, numImages);
  dim3 divBlock(blockSize * blockSize);
  dim3 divGrid((numImages + blockSize * blockSize - 1) / (blockSize * blockSize));

  unsigned char *d_input;
  cudaMalloc((void **)&d_input, numImages * imageSize * sizeof(unsigned char));

  unsigned int *d_imageSums;
  cudaMalloc((void **)&d_imageSums, numImages * sizeof(unsigned int));

  unsigned char *d_imageMeans;
  cudaMalloc((void **)&d_imageMeans, numImages * sizeof(unsigned char));

  cudaMemcpy(d_input, imagesData.data(),
             numImages * imageSize * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  sumImages<<<gridDims, blockDims>>>(d_input, d_imageSums, width, height);
  cudaDeviceSynchronize();

  divideAreas<<<divGrid, divBlock>>>(d_imageSums, d_imageMeans, numImages,
                                     width * height);
  cudaDeviceSynchronize();

  std::vector<unsigned char> hostMeans(numImages);
  cudaMemcpy(hostMeans.data(), d_imageMeans, numImages * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_imageSums);
  cudaFree(d_imageMeans);

  return hostMeans;
}