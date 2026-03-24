#include "ImageEvaluator.h"
#include <vector>

float ImageEvaluator::PSNR(ImageBase &origin, ImageBase &output) {
  double EQM = 0.0;
  int sideSize = origin.getHeight();

  for (int y = 0; y < sideSize; ++y) {
    for (int x = 0; x < sideSize; ++x) {
      int val_orig = origin[y][x];
      int val_rec = output[y][x];
      EQM += (val_orig - val_rec) * (val_orig - val_rec);
    }
  }

  EQM = EQM / (sideSize * sideSize);

  if (EQM == 0) {
    printf("PSNR = Infini (Images identiques)\n");
    return 99.0;
  } else {
    double psnr = 10.0 * log10((255.0 * 255.0) / EQM);
    printf("EQM = %f\n", EQM);
    printf("PSNR = %f dB\n", psnr);
    return (float)psnr;
  }
}

int ImageEvaluator::diffHisto(ImageBase &origin,
                              ImageBase &output) {
  int sideSize = origin.getHeight();
  std::vector<int> histoInput = std::vector<int>(256, 0);
  std::vector<int> histoOutput = std::vector<int>(256, 0);
  for (int y = 0; y < sideSize; ++y) {
    for (int x = 0; x < sideSize; ++x) {
      histoInput[imgIN[y][x]]++;
      histoOutput[imgOUT[y][x]]++;
    }
  }
  int d = 0;
  for (int i = 0; i < histoIN.size(); i++) {
    d += abs(histoIN[i] - histoOUT[i]);
  }
  return d;
}