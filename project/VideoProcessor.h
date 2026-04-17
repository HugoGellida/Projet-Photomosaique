#pragma once
#include "ImageBase.h"
#include <vector>

class VideoProcessor {
public:
  // Process a single frame using smart composition (reuses previous frame
  // composition)
  static std::vector<int>
  processFrameSmart(const std::vector<unsigned char> &currentMeans,
                    const std::vector<unsigned char> &datasetMeans,
                    std::vector<int> &prevComposition, float threshold);

  static std::vector<int>
  processFrameSmartUnique(const std::vector<unsigned char> &currentMeans,
                    const std::vector<unsigned char> &datasetMeans,
                    std::vector<int> &prevComposition, float threshold, std::vector<bool> &used);

  // GPU-accelerated version of processFrameSmart
  static ImageBase
  processFrameGPU(const unsigned char *frameData, int width, int height,
                  const std::vector<unsigned char> &datasetLocalMeans,
                  const std::vector<unsigned char> &datasetMeans,
                  std::vector<int> &prevComposition, int sideOfImage, int tileSize);

  // Process frame and return final mosaic
  static ImageBase
  processFrame(const unsigned char *frameData, int width, int height,
               const std::vector<unsigned char> &datasetLocalMeans,
               const std::vector<unsigned char> &datasetMeans,
               std::vector<int> &prevComposition, int sideOfImage);

               static ImageBase
  processFrameUnique(const unsigned char *frameData, int width, int height,
               const std::vector<unsigned char> &datasetLocalMeans,
               const std::vector<unsigned char> &datasetMeans,
               std::vector<int> &prevComposition, int sideOfImage, std::vector<bool> &used);
};
