#include "ImageBase.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

int sideOfImage = 64; // number of tiles per row/column for final image
int smallTileSizeInPixels = 64; // size of each small image tile

int requested_width = 128;  // expected width of dataset images
int requested_height = 128; // expected height of dataset images

namespace fs = std::filesystem;

// CUDA kernel: sum all pixels of each image
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
getImagesLocalMeans(std::vector<unsigned char> imageData, int width, int height,
                    int divider) {

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
std::vector<unsigned char> getImagesMeans(std::vector<unsigned char> imagesData,
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

float PSNRv1(std::vector<unsigned char> targetLocalMeans, std::vector<unsigned char> datasetMeans, std::vector<int> compositionOrder) {
    double EQM = 0.0;
    int sideSize = sqrt(compositionOrder.size());
  
    for (int y = 0; y < sideSize; ++y) {
        for (int x = 0; x < sideSize; ++x) {
              int val_orig = targetLocalMeans[y*sideSize+x];
              int val_rec  = datasetMeans[compositionOrder[y*sideSize+x]];
              EQM += (val_orig - val_rec) * (val_orig - val_rec);
        
        }
    }

    EQM = EQM / (sideSize * sideSize);

    if (EQM == 0) {
        printf("PSNR = Infini (Images identiques)\n");
        return 99.0;
    } else {
        double psnr = 10.0 * log10((255.0 * 255.0) / EQM);
        printf("EQM = %f\n", EQM);
        printf("PSNR = %f dB\n", psnr);
        return (float)psnr;
    }
}

float PSNRv2(ImageBase& imgIN, ImageBase& imgOUT) {
    double EQM = 0.0;
    int sideSize = imgIN.getHeight();
  
    for (int y = 0; y < sideSize; ++y) {
        for (int x = 0; x < sideSize; ++x) {
              int val_orig =  imgIN[y][x];
              int val_rec  =  imgOUT[y][x];
              EQM += (val_orig - val_rec) * (val_orig - val_rec);
        
        }
    }

    EQM = EQM / (sideSize * sideSize);

    if (EQM == 0) {
        printf("PSNR = Infini (Images identiques)\n");
        return 99.0;
    } else {
        double psnr = 10.0 * log10((255.0 * 255.0) / EQM);
        printf("EQM = %f\n", EQM);
        printf("PSNR = %f dB\n", psnr);
        return (float)psnr;
    }
}

double computeSSIM(const std::vector<double>& imgIN,
                   const std::vector<double>& imgOUT) {
    int N = imgIN.size();

    // Moyennes
    double m_x = 0.0, m_y = 0.0;
    for (int i = 0; i < N; i++) {
        m_x += imgIN[i];
        m_y += imgOUT[i];
    }
    m_x /= N;
    m_y /= N;

    // Variances et covariance
    double sigma_x = 0.0, sigma_y = 0.0, sigma_xy = 0.0;

    for (int i = 0; i < N; i++) {
        double dx = imgIN[i] - m_x;
        double dy = imgOUT[i] - m_y;

        sigma_x += dx * dx;
        sigma_y += dy * dy;
        sigma_xy += dx * dy;
    }

    sigma_x /= (N - 1);
    sigma_y /= (N - 1);
    sigma_xy /= (N - 1);

    // Constantes
    const double L = 255.0;
    const double C1 = (0.01 * L) * (0.01 * L);
    const double C2 = (0.03 * L) * (0.03 * L);

    // SSIM
    double numerator = (2 * m_x * m_y + C1) * (2 * sigma_xy + C2);
    double denominator = (m_x * m_x + m_y * m_y + C1) *
                         (sigma_x + sigma_y + C2);

    return numerator / denominator;
}

double MeanSSIM(ImageBase& imgIN, ImageBase& imgOUT) {

    int downscale = 4;
  int taille = imgIN.getHeight();

  std::vector<unsigned char> inputDataVecIN;
  unsigned char *inputDataIN = imgIN->getData();
  inputDataVecIN.insert(inputDataVecIN.end(), inputDataIN,
                      inputDataIN + taille * taille);

  std::vector<unsigned char> inputDataVecOUT;
  unsigned char *inputDataOUT = imgOUT->getData();
  inputDataVecOUT.insert(inputDataVecOUT.end(), inputDataOUT,
                      inputDataOUT + taille * taille);

  
    std::vector<unsigned char> imgResizeIN = getImagesLocalMeans(inputDataVecIN, imgIN.getWidth(), imgIN.getHeight(), downscale);
    std::vector<unsigned char> imgResizeOUT = getImagesLocalMeans(inputDataVecOUT, imgOUT.getWidth(), imgOUT.getHeight(), downscale);

    int gridSize = 8;
    int nbGridY = imgIN.getHeight() / gridSize / downscale;
    int nbGridX = imgIN.getWidth() / gridSize / downscale;

    double moyenne = 0.0;
    int count = 0;
    int resizedWidth = imgIN.getWidth() / downscale;

    for (int gy = 0; gy < nbGridY; gy++) {
        for (int gx = 0; gx < nbGridX; gx++) {

            std::vector<double> blockIN;
            std::vector<double> blockOUT;

            for (int y = 0; y < gridSize; y++) {
                for (int x = 0; x < gridSize; x++) {

                    int iy = gy * gridSize + y;
                    int ix = gx * gridSize + x;

                    blockIN.push_back(imgResizeIN[iy * resizedWidth + ix]);
                    blockOUT.push_back(imgResizeOUT[iy * resizedWidth + ix]);
                }
            }

            moyenne += computeSSIM(blockIN, blockOUT);
            count++;
        }
    }

    return moyenne / count;
}


int diffHisto(ImageBase& imgIN, ImageBase& imgOUT) {
  int sideSize = imgIN.getHeight();
  std::vector<int> histoIN = std::vector<int>(256,0);
  std::vector<int> histoOUT = std::vector<int>(256,0);
    for (int y = 0; y < sideSize; ++y) {
        for (int x = 0; x < sideSize; ++x) {
            histoIN[imgIN[y][x]] ++;
            histoOUT[imgOUT[y][x]] ++;
        }
      }
      int d = 0;
    for (int i = 0; i < histoIN.size(); i++){
      d += abs(histoIN[i] - histoOUT[i]); 
    }
    return d;   
}



ImageBase resizeImage(std::vector<unsigned char> image){
  int wantedSize = sideOfImage * sideOfImage * smallTileSizeInPixels * smallTileSizeInPixels;
  int sizeImage = image.size();
  int sideImage = sqrt(image.size());
  int diff = sqrt(wantedSize) / sideImage;

  ImageBase imageOUT(sqrt(wantedSize), sqrt(wantedSize), false);

  for (int y = 0 ; y < sideImage; y ++){
    for (int x = 0 ; x < sideImage; x++ ){
      for (int yi = 0 ; yi < diff; yi++ ){
        for (int xi = 0 ; xi < diff; xi++ ){
          imageOUT[yi + diff*y ][xi + diff * x] = (int) image[y*sideImage+x];

        }
      }
    }
  }
  return imageOUT;
  
}


// Compose final image from small images
ImageBase composeImg(const std::vector<unsigned char> &smallImagesData,
                     int numImages, const std::vector<int> &composition) {

  int smallImageSize = smallImagesData.size() / numImages;
  int smallImageSide = sqrt(smallImageSize);
  int gridSide = sqrt(composition.size());

  ImageBase output(gridSide * smallImageSide, gridSide * smallImageSide, false);

  for (int tileX = 0; tileX < gridSide; tileX++)
    for (int tileY = 0; tileY < gridSide; tileY++)
      for (int x = 0; x < smallImageSide; x++)
        for (int y = 0; y < smallImageSide; y++)
          output[x + tileX * smallImageSide][y + tileY * smallImageSide] =
              smallImagesData[composition[tileX * gridSide + tileY] *
                                  smallImageSize +
                              x * smallImageSide + y];

  return output;
}

// Select images allowing repeats
std::vector<int> orderImg(const std::vector<unsigned char> &targetLocalMeans,
                          const std::vector<unsigned char> &datasetMeans) {

  std::vector<int> selectedIndices;
  for (int i = 0; i < targetLocalMeans.size(); i++) {
    int bestIndex = 0;
    for (int j = 1; j < datasetMeans.size(); j++)
      if ((targetLocalMeans[i] - datasetMeans[j]) *
              (targetLocalMeans[i] - datasetMeans[j]) <
          (targetLocalMeans[i] - datasetMeans[bestIndex]) *
              (targetLocalMeans[i] - datasetMeans[bestIndex]))
        bestIndex = j;

    selectedIndices.push_back(bestIndex);
  }

  return selectedIndices;
}

// Select images without repeats
std::vector<int>
orderImgUnique(const std::vector<unsigned char> &targetLocalMeans,
               const std::vector<unsigned char> &datasetMeans) {

  std::vector<int> selectedIndices(targetLocalMeans.size(), -1);
  std::vector<bool> used(datasetMeans.size(), false);

  for (int i = 0; i < targetLocalMeans.size(); i++) {
    int bestIndex = -1;
    int bestDist = INT_MAX;

    for (int j = 0; j < datasetMeans.size(); j++) {
      if (used[j])
        continue;
      int dist = (targetLocalMeans[i] - datasetMeans[j]) *
                 (targetLocalMeans[i] - datasetMeans[j]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIndex = j;
      }
    }

    if (bestIndex != -1) {
      used[bestIndex] = true;
      selectedIndices[i] = bestIndex;
    }
  }

  return selectedIndices;
}

// Struct for prioritizing difficult zones
struct Zone {
  int index;
  int difficulty;
};

// Select images without repeats with priority
std::vector<int>
orderImgPriority(const std::vector<unsigned char> &targetLocalMeans,
                 const std::vector<unsigned char> &datasetMeans) {

  std::vector<int> selectedIndices(targetLocalMeans.size(), -1);
  std::vector<bool> used(datasetMeans.size(), false);
  std::vector<Zone> zones(targetLocalMeans.size());

  int threshold = 100;

  // Compute difficulty for each zone
  for (int i = 0; i < targetLocalMeans.size(); i++) {
    zones[i].index = i;
    int count = 0;
    for (int j = 0; j < datasetMeans.size(); j++)
      if ((targetLocalMeans[i] - datasetMeans[j]) *
              (targetLocalMeans[i] - datasetMeans[j]) <
          threshold)
        count++;
    zones[i].difficulty = count;
  }

  // Sort zones by difficulty ascending
  std::sort(zones.begin(), zones.end(), [](const Zone &a, const Zone &b) {
    return a.difficulty < b.difficulty;
  });

  // Assign images
  for (int k = 0; k < targetLocalMeans.size(); k++) {
    int zoneIdx = zones[k].index;
    unsigned char target = targetLocalMeans[zoneIdx];

    int bestIndex = -1;
    int bestDist = INT_MAX;

    for (int j = 0; j < datasetMeans.size(); j++) {
      if (used[j])
        continue;
      int dist = (target - datasetMeans[j]) * (target - datasetMeans[j]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIndex = j;
      }
    }

    if (bestIndex != -1) {
      used[bestIndex] = true;
      selectedIndices[zoneIdx] = bestIndex;
    }
  }

  // Optional improvement by swapping
  bool improved = true;
  int N = selectedIndices.size();
  while (improved) {
    improved = false;
    for (int i = 0; i < N; i++) {
      for (int j = i + 1; j < N; j++) {
        int imgA = selectedIndices[i];
        int imgB = selectedIndices[j];

        int oldCost = (targetLocalMeans[i] - datasetMeans[imgA]) *
                          (targetLocalMeans[i] - datasetMeans[imgA]) +
                      (targetLocalMeans[j] - datasetMeans[imgB]) *
                          (targetLocalMeans[j] - datasetMeans[imgB]);

        int newCost = (targetLocalMeans[i] - datasetMeans[imgB]) *
                          (targetLocalMeans[i] - datasetMeans[imgB]) +
                      (targetLocalMeans[j] - datasetMeans[imgA]) *
                          (targetLocalMeans[j] - datasetMeans[imgA]);

        if (newCost < oldCost) {
          std::swap(selectedIndices[i], selectedIndices[j]);
          improved = true;
        }
      }
    }
  }

  return selectedIndices;
}

//* ======== MAIN ========

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Wrong use" << std::endl;
    return 1;
  }

  // Load target image
  char *inputPath = argv[1];
  ImageBase *inputImage = new ImageBase();
  inputImage->load(inputPath);

  int width = inputImage->getWidth();
  int height = inputImage->getHeight();

  std::vector<unsigned char> inputDataVec;
  unsigned char *inputData = inputImage->getData();
  inputDataVec.insert(inputDataVec.end(), inputData,
                      inputData + width * height);

  // Load dataset images
  char *datasetFolder = argv[2];
  std::vector<ImageBase *> datasetImages;
  std::vector<unsigned char> datasetData;

  float totalImages = 0.f;
  for (const auto &entry : fs::directory_iterator(datasetFolder))
    if (entry.is_regular_file() && entry.path().extension() == ".pgm")
      totalImages++;

  float current = 0.f;
  for (const auto &entry : fs::directory_iterator(datasetFolder)) {
    if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
      std::string path = entry.path().string();
      ImageBase *img = new ImageBase();
      img->load(path.c_str());
      if (img->getWidth() == requested_width &&
          img->getHeight() == requested_height) {
        datasetImages.push_back(img);
        unsigned char *data = img->getData();
        datasetData.insert(datasetData.end(), data,
                           data + requested_width * requested_height);
      }
      current++;
      float percent = (current * 100) / totalImages;
      std::cout << "\rLoading images: " << percent << "% (" << current << "/"
                << totalImages << ")" << std::flush;
    }
  }
  std::cout << std::endl;

  // Compute global and local means
  std::vector<unsigned char> datasetMeans =
      getImagesMeans(datasetData, requested_width, requested_height);
  std::vector<unsigned char> targetLocalMeans =
      getImagesLocalMeans(inputDataVec, width, height, sideOfImage);
  std::vector<unsigned char> datasetLocalMeans = getImagesLocalMeans(
      datasetData, requested_width, requested_height, smallTileSizeInPixels);

  // Order images
  std::vector<int> compositionOrder = orderImgUnique(targetLocalMeans, datasetMeans);

  float psnr = PSNRv1(targetLocalMeans, datasetMeans, compositionOrder);

  // Compose final image
  ImageBase finalImage =
      composeImg(datasetLocalMeans, datasetImages.size(), compositionOrder);
  finalImage.save("out.pgm");

  ImageBase inputResize = resizeImage(inputDataVec);
  psnr = PSNRv2(inputResize, finalImage );

  std::cout << "Moyenne SSIM : "<< MeanSSIM(inputResize, finalImage) << std::endl;
  //inputResize.save("test.pgm");

  return 0;
}