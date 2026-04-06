#pragma once
#include <vector>

class ImageProcessor {
public:
  static std::vector<unsigned char>
  getGlobalMeans(const std::vector<unsigned char> &imagesData, int width,
                 int height);
  static std::vector<unsigned char>
  getLocalMeans(const std::vector<unsigned char> &imagesData, int width,
                int height, int divider);

  // GPU-accelerated smart frame processing kernel wrapper
  static void processFrameSmartGPU(const unsigned char *currentMeans,
                                   const unsigned char *datasetMeans,
                                   const int *prevComposition, int *composition,
                                   int sideOfImage, int numDatasetImages,
                                   float threshold);
};