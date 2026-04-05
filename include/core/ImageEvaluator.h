#include "core/ImageBase.h"

class ImageEvaluator {
public:
  static float PSNR(ImageBase &origin, ImageBase &output);
  static int diffHisto(ImageBase &origin, ImageBase &output);
};