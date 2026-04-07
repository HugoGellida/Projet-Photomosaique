#include "ImageComposer.h"
#include <cmath>

ImageBase ImageComposer::compose(const std::vector<unsigned char> &tilesData,
                                 int numTiles,
                                 const std::vector<int> &compositionOrder) {
  int t = tilesData.size() / numTiles;
  int tileSize = std::sqrt(t);
  int gridSide = sqrt(compositionOrder.size());

  ImageBase output(gridSide * tileSize, gridSide * tileSize, false);

  for (int tileY = 0; tileY < gridSide; tileY++)
    for (int tileX = 0; tileX < gridSide; tileX++)
      for (int y = 0; y < tileSize; y++)
        for (int x = 0; x < tileSize; x++)
          output[tileY * tileSize + y][tileX * tileSize + x] =
              tilesData[compositionOrder[tileY * gridSide + tileX] * t +
                        y * tileSize + x];
  return output;
}

void ImageComposer::composeV2(const std::vector<unsigned char> &tilesData,
                              int numTiles,
                              const std::vector<int> &compositionOrder,
                              ImageBase &output) {
  int t = tilesData.size() / numTiles;
  int tileSize = std::sqrt(t);
  int gridSide = sqrt(compositionOrder.size());

  for (int tileY = 0; tileY < gridSide; tileY++)
    for (int tileX = 0; tileX < gridSide; tileX++)
      for (int y = 0; y < tileSize; y++)
        for (int x = 0; x < tileSize; x++)
          output[tileY * tileSize + y][tileX * tileSize + x] =
              tilesData[compositionOrder[tileY * gridSide + tileX] * t +
                        y * tileSize + x];
}

ImageBase ImageComposer::composeV3(const std::vector<unsigned char> &tilesData,
                                   int numTiles,
                                   const std::vector<int> &compositionOrder,
                                   int smallTileSize, int datasetTileSize) {
  int gridSide = sqrt(compositionOrder.size());
  int mosaicWidth = gridSide * smallTileSize;
  int mosaicHeight = gridSide * smallTileSize;

  ImageBase output(mosaicWidth, mosaicHeight, false);

  for (int tileY = 0; tileY < gridSide; ++tileY) {
    for (int tileX = 0; tileX < gridSide; ++tileX) {
      int tileIndex = compositionOrder[tileY * gridSide + tileX];

      for (int y = 0; y < smallTileSize; ++y) {
        for (int x = 0; x < smallTileSize; ++x) {
          // Mapping proportionnel dataset -> imagette finale
          int srcX = int(x * float(datasetTileSize) / smallTileSize);
          int srcY = int(y * float(datasetTileSize) / smallTileSize);

          int mosaicIndex = (tileY * smallTileSize + y) * mosaicWidth +
                            (tileX * smallTileSize + x);
          int tileDataIndex = tileIndex * (datasetTileSize * datasetTileSize) +
                              srcY * datasetTileSize + srcX;

          output.getData()[mosaicIndex] = tilesData[tileDataIndex];
        }
      }
    }
  }

  return output;
}