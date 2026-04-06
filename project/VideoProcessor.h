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

  // GPU-accelerated version of processFrameSmart
  static void
  processFrameGPU(const unsigned char *frameData, int width, int height,
                  const std::vector<unsigned char> &datasetLocalMeans,
                  const std::vector<unsigned char> &datasetMeans,
                  std::vector<int> &prevComposition, ImageBase &output,
                  int sideOfImage);

  // Process frame and return final mosaic
  static ImageBase
  processFrame(const unsigned char *frameData, int width, int height,
               const std::vector<unsigned char> &datasetLocalMeans,
               const std::vector<unsigned char> &datasetMeans,
               std::vector<int> &prevComposition, int sideOfImage);
};
