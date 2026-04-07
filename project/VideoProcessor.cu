#include "ImageComposer.h"
#include "ImageProcessor.h"
#include "VideoProcessor.h"
#include <algorithm>
#include <climits>
#include <cmath>

std::vector<int> VideoProcessor::processFrameSmart(
    const std::vector<unsigned char> &currentMeans,
    const std::vector<unsigned char> &datasetMeans,
    std::vector<int> &prevComposition, float threshold) {

  std::vector<int> composition(currentMeans.size());

  // For first frame, use standard ordering
  if (prevComposition.empty() || prevComposition[0] == -1) {
    composition.resize(currentMeans.size());
    for (int i = 0; i < currentMeans.size(); i++) {
      int bestIndex = 0;
      int bestDist = INT_MAX;
      for (int j = 0; j < datasetMeans.size(); j++) {
        int dist = (currentMeans[i] - datasetMeans[j]) *
                   (currentMeans[i] - datasetMeans[j]);
        if (dist < bestDist) {
          bestDist = dist;
          bestIndex = j;
        }
      }
      composition[i] = bestIndex;
    }
    prevComposition = composition;
    return composition;
  }

  // For subsequent frames, reuse compositions within threshold
  for (int i = 0; i < currentMeans.size(); i++) {
    float diff = std::abs((float)currentMeans[i] -
                          (float)datasetMeans[prevComposition[i]]);

    if (diff < threshold) {
      // Keep previous composition for this tile
      composition[i] = prevComposition[i];
      continue;
    }

    // Find best matching image
    int bestIndex = 0;
    int bestDist = INT_MAX;

    for (int j = 0; j < datasetMeans.size(); j++) {
      int dist = (currentMeans[i] - datasetMeans[j]) *
                 (currentMeans[i] - datasetMeans[j]);

      if (dist < bestDist) {
        bestDist = dist;
        bestIndex = j;
        if (bestDist == 0)
          break;
      }
    }

    composition[i] = bestIndex;
  }

  prevComposition = composition;
  return composition;
}

ImageBase VideoProcessor::processFrame(
    const unsigned char *frameData, int width, int height,
    const std::vector<unsigned char> &datasetLocalMeans,
    const std::vector<unsigned char> &datasetMeans,
    std::vector<int> &prevComposition, int sideOfImage) {

  // Compute local means for the current frame
  std::vector<unsigned char> currentMeans = ImageProcessor::getLocalMeans(
      std::vector<unsigned char>(frameData, frameData + width * height), width,
      height, sideOfImage);

  // Process frame using smart composition
  std::vector<int> composition =
      processFrameSmart(currentMeans, datasetMeans, prevComposition, 15.0f);

  // Compose final image
  ImageBase result = ImageComposer::compose(datasetLocalMeans,
                                            datasetMeans.size(), composition);
  return result;
}

std::vector<int> VideoProcessor::processFrameSmartUnique(
    const std::vector<unsigned char> &currentMeans,
    const std::vector<unsigned char> &datasetMeans,
    std::vector<int> &prevComposition, float threshold, std::vector<bool> &used) {
  
  
  std::vector<int> composition(currentMeans.size());

  // For first frame, use standard ordering
  if (prevComposition.empty() || prevComposition[0] == -1) {
    composition.resize(currentMeans.size());
    for (int i = 0; i < currentMeans.size(); i++) {
      int bestIndex = 0;
      int bestDist = INT_MAX;
      for (int j = 0; j < datasetMeans.size(); j++) {
        if(used[j] == false){
          int dist = (currentMeans[i] - datasetMeans[j]) *
                    (currentMeans[i] - datasetMeans[j]);
          if (dist < bestDist) {
            bestDist = dist;
            bestIndex = j;
          }
        }
      }
      used[bestIndex] = true;
      composition[i] = bestIndex;
    }
    prevComposition = composition;
    return composition;
  }

  // For subsequent frames, reuse compositions within threshold
  for (int i = 0; i < currentMeans.size(); i++) {
    float diff = std::abs((float)currentMeans[i] -
                          (float)datasetMeans[prevComposition[i]]);

    if (diff < threshold) {
      // Keep previous composition for this tile
      composition[i] = prevComposition[i];
      continue;
    }

    // Find best matching image
    int bestIndex = 0;
    int bestDist = INT_MAX;

    for (int j = 0; j < datasetMeans.size(); j++) {
      if (used[j] == false){
        int dist = (currentMeans[i] - datasetMeans[j]) *
                  (currentMeans[i] - datasetMeans[j]);

        if (dist < bestDist) {
          bestDist = dist;
          bestIndex = j;
          if (bestDist == 0)
            break;
        }
      }
    }
    used[prevComposition[i]] = false;
    used[bestIndex] = true;
    composition[i] = bestIndex;
  }
  for (int z = 0; z < 3; z++){
    for (int i = 0; i < currentMeans.size(); i++) {
      if (abs(currentMeans[i] - datasetMeans[prevComposition[i]]) < threshold)
        continue;
      for (int j = i + 1; j < currentMeans.size(); j++) {
        int imgA = composition[i];
        int imgB = composition[j];

        int oldCost = (currentMeans[i] - datasetMeans[imgA]) *
                          (currentMeans[i] - datasetMeans[imgA]) +
                      (currentMeans[j] - datasetMeans[imgB]) *
                          (currentMeans[j] - datasetMeans[imgB]);

        int newCost = (currentMeans[i] - datasetMeans[imgB]) *
                          (currentMeans[i] - datasetMeans[imgB]) +
                      (currentMeans[j] - datasetMeans[imgA]) *
                          (currentMeans[j] - datasetMeans[imgA]);

        if (newCost + 500 < oldCost) {
          std::swap(composition[i], composition[j]);
        }
      }
    }
  }

  prevComposition = composition;
  return composition;
}

ImageBase VideoProcessor::processFrameUnique(
    const unsigned char *frameData, int width, int height,
    const std::vector<unsigned char> &datasetLocalMeans,
    const std::vector<unsigned char> &datasetMeans,
    std::vector<int> &prevComposition, int sideOfImage, std::vector<bool> &used) {

  // Compute local means for the current frame
  std::vector<unsigned char> currentMeans = ImageProcessor::getLocalMeans(
      std::vector<unsigned char>(frameData, frameData + width * height), width,
      height, sideOfImage);

  // Process frame using smart composition
  std::vector<int> composition =
      processFrameSmartUnique(currentMeans, datasetMeans, prevComposition, 15.0f, used);

  // Compose final image
  ImageBase result = ImageComposer::compose(datasetLocalMeans,
                                            datasetMeans.size(), composition);
  return result;
}

void VideoProcessor::processFrameGPU(
    const unsigned char *frameData, int width, int height,
    const std::vector<unsigned char> &datasetLocalMeans,
    const std::vector<unsigned char> &datasetMeans,
    std::vector<int> &prevComposition, ImageBase &output, int sideOfImage) {

  // Compute local means for the current frame
  std::vector<unsigned char> currentMeans = ImageProcessor::getLocalMeans(
      std::vector<unsigned char>(frameData, frameData + width * height), width,
      height, sideOfImage);

  int numTiles = currentMeans.size();
  int numDatasetImages = datasetMeans.size();

  // Initialize for first frame
  if (prevComposition.empty() || prevComposition[0] == -1) {
    prevComposition.assign(numTiles, 0);
    for (int i = 0; i < numTiles; i++) {
      int bestIndex = 0;
      int bestDist = INT_MAX;
      for (int j = 0; j < numDatasetImages; j++) {
        int dist = (currentMeans[i] - datasetMeans[j]) *
                   (currentMeans[i] - datasetMeans[j]);
        if (dist < bestDist) {
          bestDist = dist;
          bestIndex = j;
        }
      }
      prevComposition[i] = bestIndex;
    }
  }

  // Run GPU kernel for smart composition
  std::vector<int> composition(numTiles);
  ImageProcessor::processFrameSmartGPU(
      currentMeans.data(), datasetMeans.data(), prevComposition.data(),
      composition.data(), sideOfImage, numDatasetImages, 15.f);

  prevComposition = composition;

  // Compose the final mosaic image
  ImageComposer::composeV2(datasetLocalMeans, numDatasetImages, composition,
                           output);
}
