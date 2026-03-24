#include "ImageBase.h"
#include "DatasetManager.h"
#include "ImageProcessor.h"
#include "ImageComposer.h"
#include "ImageOrdering.h"
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

    ImageBase finalImage = ImageComposer::compose(datasetLocalMeans, dataset.getImages().size(), compositionOrder);
    finalImage.save("./Results/out.pgm");

    return 0;
}