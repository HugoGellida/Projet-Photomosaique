#include "Utils.h"
#include <cmath>

ImageBase Utils::resizeImage(std::vector<unsigned char> image, int imagesPerSide, int imagettesSize){
  int wantedSize = imagesPerSide * imagesPerSide * imagettesSize * imagettesSize;
  int sizeImage = image.size();
  int sideImage = std::sqrt(image.size());
  int diff = std::sqrt(wantedSize) / sideImage;

  ImageBase imageOUT(sqrt(wantedSize), sqrt(wantedSize), false);

  for (int y = 0 ; y < sideImage; y ++){
    for (int x = 0 ; x < sideImage; x++ ){
      for (int yi = 0 ; yi < diff; yi++ ){
        for (int xi = 0 ; xi < diff; xi++ ){
          imageOUT[yi + diff*y ][xi + diff * x] = (int) image[y*sideImage+x];

        }
      }
    }
  }
  return imageOUT;
  
}