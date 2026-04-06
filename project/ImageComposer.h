#pragma once
#include "ImageBase.h"
#include <vector>

class ImageComposer {
public:
  static ImageBase compose(const std::vector<unsigned char> &tilesData,
                           int numTiles,
                           const std::vector<int> &compositionOrder);

  // Compose directly into an existing ImageBase (for efficiency)
  static void composeV2(const std::vector<unsigned char> &tilesData,
                        int numTiles, const std::vector<int> &compositionOrder,
                        ImageBase &output);
};