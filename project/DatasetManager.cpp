#include "DatasetManager.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

DatasetManager::DatasetManager(int exceptedWidth, int exceptedHeight): width(exceptedWidth), height(exceptedHeight) {}

void DatasetManager::loadFromFolder(const std::string &folderPath){
    float totalImages = 0.f;
    for (const auto &entry : fs::directory_iterator(folderPath))
        if (entry.is_regular_file() && entry.path().extension() == ".pgm")
        totalImages++;

    float current = 0.f;
    images.clear();
    for (const auto &entry : fs::directory_iterator(folderPath)){
        if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
            ImageBase *img = new ImageBase();
            img->load(entry.path().string().c_str());
            if (img->getWidth() == width && img->getHeight() == height) {
                images.push_back(img);
            } else {
                delete img;
            }
        }
        current++;
        float percent = (current * 100) / totalImages;
        std::cout << "\rLoading images: " << percent << "% (" << current << "/" << totalImages << ")" << std::flush;
    }
    std::cout << "Loaded images: " << images.size() << std::endl;
}

const std::vector<ImageBase *> &DatasetManager::getImages() const {
    return images;
}

std::vector<unsigned char> DatasetManager::getConcatenatedData() const {
    std::vector<unsigned char> allData;
    for (auto img: images){
        allData.insert(allData.end(), img->getData(), img->getData() + width * height);
    }
    return allData;
}