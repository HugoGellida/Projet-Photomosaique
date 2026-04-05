#include "core/ImageBase.h"
#include <vector>

class Utils {
public:
  static ImageBase resizeImage(std::vector<unsigned char> image,
                               int imagesPerSide, int imagettesSize);
};