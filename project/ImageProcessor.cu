#include "ImageProcessor.h"
#include <limits>
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
std::vector<unsigned char>
ImageProcessor::getLocalMeans(const std::vector<unsigned char> &imageData,
                              int width, int height, int divider) {

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
  cudaMalloc((void **)&d_areaMeans,
             numImages * numAreas * sizeof(unsigned char));
  cudaMemcpy(d_input, imageData.data(),
             numImages * imageSize * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  areaSums<<<gridDims, blockDims>>>(d_input, d_areaSums, width, height,
                                    divider);
  cudaDeviceSynchronize();
  divideAreas<<<divGrid, divBlock>>>(d_areaSums, d_areaMeans, numAreas,
                                     pixelsPerArea);
  cudaDeviceSynchronize();
  std::vector<unsigned char> hostMeans(numImages * numAreas);
  cudaMemcpy(hostMeans.data(), d_areaMeans,
             numImages * numAreas * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_areaSums);
  cudaFree(d_areaMeans);

  return hostMeans;
}

// Compute mean intensity for entire images
std::vector<unsigned char>
ImageProcessor::getGlobalMeans(const std::vector<unsigned char> &imagesData,
                               int width, int height) {

  size_t numImages = imagesData.size() / (width * height);
  size_t imageSize = width * height;
  int blockSize = 16;
  dim3 blockDims(blockSize, blockSize);
  dim3 gridDims((width + blockSize - 1) / blockSize,
                (height + blockSize - 1) / blockSize, numImages);
  dim3 divBlock(blockSize * blockSize);
  dim3 divGrid((numImages + blockSize * blockSize - 1) /
               (blockSize * blockSize));

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

__global__ void processFrameSmartKernel(const unsigned char *currentMeans,
                                        const unsigned char *datasetMeans,
                                        const int *prevComposition,
                                        int *composition, int sideOfImage,
                                        int numDatasetImages, float threshold) {
  int tileX = blockIdx.x * blockDim.x + threadIdx.x;
  int tileY = blockIdx.y * blockDim.y + threadIdx.y;

  if (tileX >= sideOfImage || tileY >= sideOfImage)
    return;

  int i = tileY * sideOfImage + tileX;

  float diff =
      fabsf((float)currentMeans[i] - (float)datasetMeans[prevComposition[i]]);
  if (diff < threshold) {
    composition[i] = prevComposition[i];
    return;
  }

  int bestIndex = 0;
  int bestDist = INT_MAX;

  for (int j = 0; j < numDatasetImages; j++) {
    int d = ((int)currentMeans[i] - (int)datasetMeans[j]);
    int dist = d * d;
    if (dist < bestDist) {
      bestDist = dist;
      bestIndex = j;
      if (bestDist == 0)
        break;
    }
  }

  composition[i] = bestIndex;
}

void ImageProcessor::processFrameSmartGPU(const unsigned char *currentMeans,
                                          const unsigned char *datasetMeans,
                                          const int *prevComposition,
                                          int *composition, int sideOfImage,
                                          int numDatasetImages,
                                          float threshold) {

  unsigned char *d_currentMeans, *d_datasetMeans;
  int *d_prevComposition, *d_composition;

  int numTiles = sideOfImage * sideOfImage;

  cudaMalloc(&d_currentMeans, numTiles * sizeof(unsigned char));
  cudaMalloc(&d_datasetMeans, numDatasetImages * sizeof(unsigned char));
  cudaMalloc(&d_prevComposition, numTiles * sizeof(int));
  cudaMalloc(&d_composition, numTiles * sizeof(int));

  cudaMemcpy(d_currentMeans, currentMeans, numTiles * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_datasetMeans, datasetMeans,
             numDatasetImages * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prevComposition, prevComposition, numTiles * sizeof(int),
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((sideOfImage + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (sideOfImage + threadsPerBlock.y - 1) / threadsPerBlock.y);

  processFrameSmartKernel<<<numBlocks, threadsPerBlock>>>(
      d_currentMeans, d_datasetMeans, d_prevComposition, d_composition,
      sideOfImage, numDatasetImages, threshold);
  cudaDeviceSynchronize();

  cudaMemcpy(composition, d_composition, numTiles * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaFree(d_currentMeans);
  cudaFree(d_datasetMeans);
  cudaFree(d_prevComposition);
  cudaFree(d_composition);
}

float ImageProcessor::computeEQM(const std::vector<unsigned char> &image1,
                                 const std::vector<unsigned char> &image2) {
  if (image1.size() != image2.size())
    return -1.0f;
  double eqm = 0.0;
  for (size_t k = 0; k < image1.size(); ++k) {
    int val1 = image1[k];
    int val2 = image2[k];
    eqm += (val1 - val2) * (val1 - val2);
  }
  eqm /= image1.size();
  return static_cast<float>(eqm);
}

std::vector<float>
ImageProcessor::computeEQMList(const std::vector<unsigned char> &target,
                               const std::vector<unsigned char> &datasetData,
                               int width, int height) {
  int numImages = datasetData.size() / (width * height);
  std::vector<float> eqmList;
  int imageSize = width * height;
  for (int i = 0; i < numImages; ++i) {
    std::vector<unsigned char> datasetImage(datasetData.begin() + i * imageSize,
                                            datasetData.begin() +
                                                (i + 1) * imageSize);
    float eqm = computeEQM(target, datasetImage);
    eqmList.push_back(eqm);
  }
  return eqmList;
}

// Compute EQM for each zone against each dataset image
std::vector<int>
ImageProcessor::computeEQMPerZone(const std::vector<unsigned char> &targetImage,
                                  const std::vector<unsigned char> &datasetData,
                                  int imageWidth, int imageHeight,
                                  int numZonesPerSide) {
  int zoneWidth = imageWidth / numZonesPerSide;
  int zoneHeight = imageHeight / numZonesPerSide;
  int numZones = numZonesPerSide * numZonesPerSide;
  int numDatasetImages = datasetData.size() / (imageWidth * imageHeight);
  int imageSize = imageWidth * imageHeight;

  std::vector<int> bestMatches(numZones);

  // For each zone
  for (int zoneY = 0; zoneY < numZonesPerSide; ++zoneY) {
    for (int zoneX = 0; zoneX < numZonesPerSide; ++zoneX) {
      int zoneIndex = zoneY * numZonesPerSide + zoneX;

      // Extract zone from target image
      std::vector<unsigned char> zone(zoneWidth * zoneHeight);
      for (int y = 0; y < zoneHeight; ++y) {
        for (int x = 0; x < zoneWidth; ++x) {
          int targetPixelX = zoneX * zoneWidth + x;
          int targetPixelY = zoneY * zoneHeight + y;
          zone[y * zoneWidth + x] =
              targetImage[targetPixelY * imageWidth + targetPixelX];
        }
      }

      // Find best matching dataset image for this zone
      int bestIndex = 0;
      float minEQM = std::numeric_limits<float>::max();

      for (int imgIdx = 0; imgIdx < numDatasetImages; ++imgIdx) {
        // Extract same zone from dataset image
        std::vector<unsigned char> datasetZone(zoneWidth * zoneHeight);
        for (int y = 0; y < zoneHeight; ++y) {
          for (int x = 0; x < zoneWidth; ++x) {
            int datasetPixelX = zoneX * zoneWidth + x;
            int datasetPixelY = zoneY * zoneHeight + y;
            datasetZone[y * zoneWidth + x] =
                datasetData[imgIdx * imageSize + datasetPixelY * imageWidth +
                            datasetPixelX];
          }
        }

        // Compute EQM between zones
        float eqm = computeEQM(zone, datasetZone);
        if (eqm < minEQM) {
          minEQM = eqm;
          bestIndex = imgIdx;
        }
      }

      bestMatches[zoneIndex] = bestIndex;
    }
  }

  return bestMatches;
}