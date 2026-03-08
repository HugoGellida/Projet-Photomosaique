#include "ImageBase.h"
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

int sideOfImage = 64;  //nombre d'imagettes utilisés par ligne et colonne pour l'image finale
int sideOfSmallImagesInPixel = 64; // taille de l'imagette pour l'image final

// important à changer pour le programme
int requested_width = 128;
int requested_height = 128; //! Ce qui est dit: les images du dataset ont des
                              //! tailles différentes. Ce ne sera pas le cas

namespace fs = std::filesystem;

__global__ void sumImages(unsigned char *d_in, unsigned int *d_out, int width,
                          int height) {
  __shared__ unsigned int sharedSum[16][16]; // shared with the rest of the
                                             // block

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int z = blockIdx.z;

  if (x < width && y < height) {
    int imgSize = width * height;
    int index = z * imgSize + y * width + x;

    sharedSum[ty][tx] = d_in[index];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tx < stride) {
        sharedSum[ty][tx] += sharedSum[ty][tx + stride];
      }
      __syncthreads();
    }

    if (tx == 0) {
      for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
        if (ty < stride) {
          sharedSum[ty][0] += sharedSum[ty + stride][0];
        }
        __syncthreads();
      }

      if (ty == 0) {
        atomicAdd(&d_out[z], sharedSum[0][0]);
      }
    }
  }
}

__global__ void areaSums(unsigned char *d_in, unsigned int *d_out, int width,
                         int height, int divider) {
  __shared__ unsigned int sharedSum[16][16];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int z = blockIdx.z;
  if (x < width && y < height) {

    int imgSize = width * height;
    int index = z * imgSize + y * width + x;

    sharedSum[ty][tx] = d_in[index];
    __syncthreads();

    int areaWidth = width / divider;
    int areaHeight = height / divider;
    int areaX = x / areaWidth;
    int areaY = y / areaHeight;

    for (int stride = 1; stride < min(blockDim.x, areaWidth); stride *= 2) {
      if (tx % (stride * 2) == 0) {
        sharedSum[ty][tx] += sharedSum[ty][tx + stride];
      }
      __syncthreads();
    }

    if (tx % areaWidth == 0) {
      for (int stride = 1; stride < min(blockDim.y, areaHeight); stride *= 2) {
        if (ty % (stride * 2) == 0) {
          sharedSum[ty][tx] += sharedSum[ty + stride][tx];
        }
        __syncthreads();
      }

      if (ty % areaHeight == 0) {
        atomicAdd(&d_out[divider * divider * z + areaY * divider + areaX],
                  sharedSum[ty][tx]);
      }
    }
  }
}

__global__
void division(unsigned int* d_in,
              unsigned char* d_out,
              int areaNbr,
              int localSize)
{
    int img = blockIdx.y;

    int area = blockIdx.x * blockDim.x + threadIdx.x;

    if (area >= areaNbr)
        return;

    int index = img * areaNbr + area;

    d_out[index] = d_in[index] / localSize;
}

std::vector<unsigned char>
getImagesLocalMeans(std::vector<unsigned char> imageChar, int width, int height,
                    int divider) {
  size_t imgSize = width * height;
  size_t localSize = (width / divider) * (height / divider);
  size_t areaNbr = divider * divider;

  int imgNbr = imageChar.size() / (width * height);

  int bSize = 16;
  dim3 sumBlockSize(bSize, bSize);
  dim3 sumGridSize((width + bSize - 1) / bSize, (height + bSize - 1) / bSize,
                   imgNbr);

  int threadsPerBlock = 256;

  dim3 meanBlockSize(threadsPerBlock);
  dim3 meanGridSize(
      (areaNbr + threadsPerBlock - 1) / threadsPerBlock,
      imgNbr
  );

  unsigned char *d_in;
  cudaMalloc((void **)&d_in, imgNbr * imgSize * sizeof(unsigned char));
  unsigned int *d_out_tot;
  cudaMalloc((void **)&d_out_tot, imgNbr * areaNbr * sizeof(unsigned int));
  unsigned char *d_out_mean;
  cudaMalloc((void **)&d_out_mean, imgNbr * areaNbr * sizeof(unsigned char));

  cudaMemcpy(d_in, imageChar.data(), imgNbr * imgSize * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  areaSums<<<sumGridSize, sumBlockSize>>>(d_in, d_out_tot, width, height, divider);
  cudaDeviceSynchronize();

  division<<<meanGridSize, meanBlockSize>>>(d_out_tot ,d_out_mean, areaNbr, localSize);
  cudaDeviceSynchronize();

  std::vector<unsigned char> hostOut(imgNbr * areaNbr);
  cudaMemcpy(hostOut.data(), d_out_mean,
             imgNbr * areaNbr * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  std::vector<unsigned int> testData(imgNbr * areaNbr);
  cudaMemcpy(testData.data(), d_out_tot,
             imgNbr * areaNbr * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out_tot);
  cudaFree(d_out_mean);
  return hostOut;
}

std::vector<unsigned char> getImagesMeans(std::vector<unsigned char> imagesChar,
                                          int width, int height) {
  size_t imgNbr = imagesChar.size() / (width * height);
  size_t imgSize = width * height;
  size_t totalSize = imgSize * imgNbr;

  int bSize = 16;
  dim3 sumBlockSize(bSize, bSize);
  dim3 sumGridSize((width + bSize - 1) / bSize, (height + bSize - 1) / bSize,
                   imgNbr);

  dim3 meanBlockSize(bSize * bSize);
  dim3 meanGridSize((imgNbr + bSize * bSize - 1) / (bSize * bSize));

  unsigned char *d_in;
  cudaMalloc((void **)&d_in, totalSize * sizeof(unsigned char));
  unsigned int *d_out_tot;
  cudaMalloc((void **)&d_out_tot, imgNbr * sizeof(unsigned int));
  unsigned char *d_out_mean;
  cudaMalloc((void **)&d_out_mean, imgNbr * sizeof(unsigned char));

  cudaMemcpy(d_in, imagesChar.data(), totalSize * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  sumImages<<<sumGridSize, sumBlockSize>>>(d_in, d_out_tot, width, height);
  cudaDeviceSynchronize();

  division<<<meanGridSize, meanBlockSize>>>(d_out_tot, d_out_mean, imgNbr,
                                            width * height);
  cudaDeviceSynchronize();

  std::vector<unsigned char> hostOut(imgNbr);
  cudaMemcpy(hostOut.data(), d_out_mean, imgNbr * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out_tot);
  cudaFree(d_out_mean);
  return hostOut;
}

ImageBase composeImg(const std::vector<unsigned char> &imagesChar, int imgNbr,
                     const std::vector<int> &composition) {
  int smallImgSize = imagesChar.size() / imgNbr;
  int smallImgSide = sqrt(smallImgSize);
  int sideSize = sqrt(composition.size());

  ImageBase imOut =
      ImageBase(sideSize * smallImgSide, sideSize * smallImgSide, false);

  for (int offX = 0; offX < sideSize; offX++)
    for (int offY = 0; offY < sideSize; offY++) {
      for (int x = 0; x < smallImgSide; x++) {
        for (int y = 0; y < smallImgSide; y++) {
          if (offX * sideSize + offY > composition.size()) std::cout << "pas normal normal" << std::endl;
          imOut[x + (offX * smallImgSide)][y + (offY * smallImgSide)] =
              imagesChar[composition[offX * sideSize + offY] * smallImgSize +
                         x * smallImgSide + y];
        }
      }
    }
  return imOut;
}

std::vector<int> orderImg(const std::vector<unsigned char> &imIn,
                          const std::vector<unsigned char> &imMeans) {
  std::vector<int> outValues = std::vector<int>();
  std::vector<bool> used = std::vector(imMeans.size(), false);
  for (int i = 0; i < imIn.size(); i++) {
    int bestIndex = -1;
    int bestDist = INT_MAX;
    for (int j = 0; j < imMeans.size(); j++) {
      if (used[j]) continue;
      int dist = (imIn[i] - imMeans[j]) * (imIn[i] - imMeans[j]);


      if (dist < bestDist) {
        bestIndex = j;
        bestDist = dist;
      }
    }
    if (bestIndex == -1) {
            std::cerr << "Plus d'images disponibles !" << std::endl;
            break;
    }
    used[bestIndex] = true;
    outValues.push_back(bestIndex);
  }
  return outValues;
} // faire cuda, et utilisation d'une seule image

//* ======== MAIN ========

int main(int argc, char **argv) {
  // ./main <image> <dossier des images>
  if (argc != 3) {
    std::cout << "Wrong use" << std::endl;
    return 1;
  }
  // Chargement de l'image cible
  char *cImageIn = argv[1];
  ImageBase *input = new ImageBase();
  input->load(cImageIn);
  int width = input->getWidth();
  int height = input->getHeight();

  std::vector<unsigned char> inputChar;
  unsigned char *inputData = input->getData();
  inputChar.insert(inputChar.end(), inputData, inputData + width * height);

  // Preparation de la liste d'image de traitement
  char *folderPath = argv[2];
  std::vector<ImageBase *> images = std::vector<ImageBase *>();
  std::vector<unsigned char> imagesChar;
  

  float totalImages = 0.f;
  for (const auto &entry : fs::directory_iterator(folderPath)) {
    if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
      totalImages++;
    }
  }

  // Extraction des images adaptées
  float current = 0.f;
  for (const auto &entry : fs::directory_iterator(folderPath)) {
    if (entry.is_regular_file()) {
      std::string path = entry.path().string();
      if (entry.path().extension() ==
          ".pgm") { //! Attention: seuls les pgms sont autorisées, l'extraction
                    //! retourne des fichiers jpg!

        // Chargement de l'image du dataset
        ImageBase *img = new ImageBase();
        img->load(path.c_str());
        int img_height = img->getHeight();
        int img_width = img->getWidth();
        if (img_height == requested_height && img_width == requested_width) {
          images.push_back(img);
          unsigned char *data = img->getData();
          imagesChar.insert(imagesChar.end(), data,
                            data + requested_width * requested_height);
        }
        current++;
        float percent = (current * 100) / totalImages;
        std::cout << "\rLoading images: " << percent << "% (" << current << "/"
                  << totalImages << ")" << std::flush;
      }
    }
  }

  std::cout << std::endl;

  std::vector<unsigned char> imgsMeans =
      getImagesMeans(imagesChar, requested_width, requested_height);

  std::vector<unsigned char> imInLocalMeans =
      getImagesLocalMeans(inputChar, width, height, sideOfImage); // les moyennes de toutes les parties de l'image d'entrée

  std::vector<unsigned char> imgsLocalMeans =
      getImagesLocalMeans(imagesChar, requested_width, requested_height, sideOfSmallImagesInPixel); // les images resize a la suite

      
  std::vector<int> order = orderImg(imInLocalMeans, imgsMeans);

  // ImageBase imOut = ImageBase(width, height, false);
  // for(int x = 0; x < width; x ++){
  //     for(int y = 0; y < height; y++){
  //         int meanX = x/(height/divider);
  //         int meanY = y/(width/divider);
  //         int mean = imInLocalMeans[meanX*divider+meanY];
  //         imOut[x][y] = mean;
  //     }
  // }
  ImageBase imOut = composeImg(imgsLocalMeans, images.size(), order);
  char cImageOut[250] = "out.pgm";
  imOut.save(cImageOut);
  return 0;
}