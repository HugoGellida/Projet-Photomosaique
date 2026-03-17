#pragma once
#include <vector>

class ImageProcessor {
    public:

    static std::vector<unsigned char> getGlobalMeans(const std::vector<unsigned char>& imagesData, int width, int height);
    static std::vector<unsigned char> getLocalMeans(const std::vector<unsigned char>& imagesData, int width, int height, int divider);
}