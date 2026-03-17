#pragma once
#include "ImageBase.h"
#include <vector>

class ImageComposer {
    public:

    static ImageBase compose(const std::vector<unsigned char> &tilesData, int numTiles, const std::vector<int> &compositionOrder);
}