#include "DatasetManager.h"
#include "ImageBase.h"
#include "ImageComposer.h"
#include "ImageEvaluator.h"
#include "ImageOrdering.h"
#include "ImageProcessor.h"
#include "Utils.h"
#include "VideoProcessor.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Video processing constants
const int VIDEO_WIDTH_INPUT = 512;
const int VIDEO_HEIGHT_INPUT = 512;
const int SIDE_OF_IMAGE = 128;
const int SMALL_TILE_SIZE = 16;
const int VIDEO_WIDTH_OUTPUT = SIDE_OF_IMAGE * SMALL_TILE_SIZE;
const int VIDEO_HEIGHT_OUTPUT = SIDE_OF_IMAGE * SMALL_TILE_SIZE;
const int FPS = 30;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0]
              << " <input.pgm|input.mp4> <dataset_folder>\n";
    return 1;
  }

  int requestedSize = 128;
  int imagesPerSide = SIDE_OF_IMAGE;
  int imagettesSize = SMALL_TILE_SIZE;

  DatasetManager dataset(requestedSize, requestedSize);
  dataset.loadFromFolder(argv[2]);

  if (dataset.getImages().empty()) {
    std::cerr << "Error: Dataset is empty\n";
    return 1;
  }

  auto datasetData = dataset.getConcatenatedData();
  auto datasetMeans =
      ImageProcessor::getGlobalMeans(datasetData, requestedSize, requestedSize);
  auto datasetLocalMeans = ImageProcessor::getLocalMeans(
      datasetData, requestedSize, requestedSize, imagettesSize);

  std::string inputPath = argv[1];
  std::string ext = inputPath.substr(inputPath.find_last_of(".") + 1);

  if (ext == "mp4" || ext == "avi" || ext == "mov") {
    // Video processing
    std::string ffmpegReadCmd = "ffmpeg -i " + inputPath +
                                " -f rawvideo -pix_fmt gray -s " +
                                std::to_string(VIDEO_WIDTH_INPUT) + "x" +
                                std::to_string(VIDEO_HEIGHT_INPUT) + " -r " +
                                std::to_string(FPS) + " -";

    FILE *ffmpegRead = popen(ffmpegReadCmd.c_str(), "r");
    if (!ffmpegRead) {
      std::cerr << "Failed to open ffmpeg input\n";
      return 1;
    }

    std::string ffmpegWriteCmd = "ffmpeg -y -f rawvideo -pix_fmt gray -s " +
                                 std::to_string(VIDEO_WIDTH_OUTPUT) + "x" +
                                 std::to_string(VIDEO_HEIGHT_OUTPUT) + " -r " +
                                 std::to_string(FPS) +
                                 " -i - -c:v libx264 -crf 28 output.mp4";

    FILE *ffmpegWrite = popen(ffmpegWriteCmd.c_str(), "w");
    if (!ffmpegWrite) {
      std::cerr << "Failed to open ffmpeg output\n";
      pclose(ffmpegRead);
      return 1;
    }

    size_t frameSize = VIDEO_WIDTH_OUTPUT * VIDEO_HEIGHT_OUTPUT;
    size_t frameInputSize = VIDEO_WIDTH_INPUT * VIDEO_HEIGHT_INPUT;
    ImageBase mosaic(VIDEO_WIDTH_OUTPUT, VIDEO_HEIGHT_OUTPUT, false);
    std::vector<unsigned char> buffer(frameInputSize);
    std::vector<int> prevComposition(SIDE_OF_IMAGE * SIDE_OF_IMAGE, -1);

    size_t frameIndex = 0;
    while (fread(buffer.data(), 1, frameInputSize, ffmpegRead) ==
           frameInputSize) {
      // Process frame
      ImageBase mosaicCPU = VideoProcessor::processFrame(
          buffer.data(), VIDEO_WIDTH_INPUT, VIDEO_HEIGHT_INPUT,
          datasetLocalMeans, datasetMeans, prevComposition, SIDE_OF_IMAGE);

      // Write to output
      size_t written = fwrite(mosaicCPU.getData(), 1, frameSize, ffmpegWrite);
      if (written != frameSize) {
        std::cerr << "\nWarning: incomplete write on frame " << frameIndex
                  << std::endl;
      }

      std::cout << "\rProcessing frame " << frameIndex++ << std::flush;
    }

    pclose(ffmpegRead);
    pclose(ffmpegWrite);

    std::cout << "\nDone. Video written to output.mp4\n";

  } else {
    // Image processing
    ImageBase targetImage;
    targetImage.load(argv[1]);

    auto targetLocalMeans = ImageProcessor::getLocalMeans(
        {targetImage.getData(),
         targetImage.getData() +
             targetImage.getWidth() * targetImage.getHeight()},
        targetImage.getWidth(), targetImage.getHeight(), imagesPerSide);

    auto compositionOrder =
        ImageOrdering::orderAllowRepeats(targetLocalMeans, datasetMeans);

    ImageBase result = ImageComposer::compose(
        datasetLocalMeans, dataset.getImages().size(), compositionOrder);

    ImageBase resizedOrigin = Utils::resizeImage(
        {targetImage.getData(),
         targetImage.getData() +
             targetImage.getWidth() * targetImage.getHeight()},
        imagesPerSide, imagettesSize);

    float PSNR = ImageEvaluator::PSNR(resizedOrigin, result);
    int diff = ImageEvaluator::diffHisto(resizedOrigin, result);

    std::cout << PSNR << " " << diff << std::endl;

    result.save("./Results/out.pgm");
  }

  return 0;
}