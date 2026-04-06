#include "core/ImageComposer.h"
#include <cmath>

ImageBase ImageComposer::compose(const std::vector<unsigned char> &tilesData,
                                 int numTiles,
                                 const std::vector<int> &compositionOrder) {
  int t = tilesData.size() / numTiles;
  int tileSize = std::sqrt(t);
  int gridSide = sqrt(compositionOrder.size());

  ImageBase output(gridSide * tileSize, gridSide * tileSize, false);

  for (int tileX = 0; tileX < gridSide; tileX++)
    for (int tileY = 0; tileY < gridSide; tileY++)
      for (int x = 0; x < tileSize; x++)
        for (int y = 0; y < tileSize; y++)
          output[x + tileX * tileSize][y + tileY * tileSize] =
              tilesData[compositionOrder[tileX * gridSide + tileY] * t +
                        x * tileSize + y];
  return output;
}

void ImageComposer::composeV2(const std::vector<unsigned char> &tilesData,
                              int numTiles,
                              const std::vector<int> &compositionOrder,
                              ImageBase &output) {
  int t = tilesData.size() / numTiles;
  int tileSize = std::sqrt(t);
  int gridSide = sqrt(compositionOrder.size());

  for (int tileX = 0; tileX < gridSide; tileX++)
    for (int tileY = 0; tileY < gridSide; tileY++)
      for (int x = 0; x < tileSize; x++)
        for (int y = 0; y < tileSize; y++)
          output[x + tileX * tileSize][y + tileY * tileSize] =
              tilesData[compositionOrder[tileX * gridSide + tileY] * t +
                        x * tileSize + y];
}