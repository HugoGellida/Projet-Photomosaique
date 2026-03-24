#include "ImageBase.h"
#include "DatasetManager.h"
#include "ImageProcessor.h"
#include "ImageComposer.h"
#include "ImageOrdering.h"
#include "ImageEvaluator.h"
#include "Utils.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <target.pgm> <dataset_folder>\n";
        return 1;
    }

    int requestedSize = 128;
    int imagesPerSide = 64;
    int imagettesSize = 16;

    ImageBase targetImage;
    targetImage.load(argv[1]);

    DatasetManager dataset(requestedSize, requestedSize);
    dataset.loadFromFolder(argv[2]);

    auto datasetData = dataset.getConcatenatedData();

    auto datasetMeans = ImageProcessor::getGlobalMeans(datasetData, requestedSize, requestedSize);
    auto targetLocalMeans = ImageProcessor::getLocalMeans(
        {targetImage.getData(), targetImage.getData() + targetImage.getWidth() * targetImage.getHeight()},
        targetImage.getWidth(), targetImage.getHeight(), imagesPerSide);

    auto datasetLocalMeans = ImageProcessor::getLocalMeans(datasetData, requestedSize, requestedSize, imagettesSize);

    auto compositionOrder = ImageOrdering::orderAllowRepeats(targetLocalMeans, datasetMeans);

    ImageBase result = ImageComposer::compose(datasetLocalMeans, dataset.getImages().size(), compositionOrder);

    ImageBase resizedOrigin = Utils::resizeImage({targetImage.getData(), targetImage.getData() + targetImage.getWidth() * targetImage.getHeight()}, imagesPerSide, imagettesSize);

    float PSNR = ImageEvaluator::PSNR(resizedOrigin, result);
    int diff = ImageEvaluator::diffHisto(resizedOrigin, result);

    std::cout << PSNR << " " << diff << std::endl;

    result.save("./Results/out.pgm");

    return 0;
}