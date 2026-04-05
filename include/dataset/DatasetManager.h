#pragma once
#include "core/ImageBase.h"
#include <string>
#include <vector>

class DatasetManager {
private:
  int width;
  int height;
  std::vector<ImageBase *> images;

public:
  DatasetManager(int exceptedWidth, int exceptedHeight);
  void loadFromFolder(const std::string &folderPath);
  const std::vector<ImageBase *> &getImages() const;
  std::vector<unsigned char> getConcatenatedData() const;
};