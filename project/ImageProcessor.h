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

  static float computeEQM(const std::vector<unsigned char> &image1,
                          const std::vector<unsigned char> &image2);

  static std::vector<float>
  computeEQMList(const std::vector<unsigned char> &target,
                 const std::vector<unsigned char> &datasetData, int width,
                 int height);

  // Compute EQM for each zone against each dataset image
  // Returns: vector of best matching dataset image index for each zone
  static std::vector<int>
  computeEQMPerZone(const std::vector<unsigned char> &targetImage,
                    const std::vector<unsigned char> &datasetData,
                    int imageWidth, int imageHeight, int numZonesPerSide);

  // GPU-accelerated smart frame processing kernel wrapper
  static void processFrameSmartGPU(const unsigned char *currentMeans,
                                   const unsigned char *datasetMeans,
                                   const int *prevComposition, int *composition,
                                   int sideOfImage, int numDatasetImages,
                                   float threshold);
};