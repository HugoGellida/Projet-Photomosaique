#include "core/ImageOrdering.h"
#include <algorithm>
#include <climits>

std::vector<int>
ImageOrdering::orderAllowRepeats(const std::vector<unsigned char> &target,
                                 const std::vector<unsigned char> &dataset) {

  std::vector<int> selectedIndices;
  for (int i = 0; i < target.size(); i++) {
    int bestIndex = 0;
    for (int j = 1; j < dataset.size(); j++)
      if ((target[i] - dataset[j]) * (target[i] - dataset[j]) <
          (target[i] - dataset[bestIndex]) * (target[i] - dataset[bestIndex]))
        bestIndex = j;

    selectedIndices.push_back(bestIndex);
  }

  return selectedIndices;
}

std::vector<int>
ImageOrdering::orderUnique(const std::vector<unsigned char> &target,
                           const std::vector<unsigned char> &dataset) {

  std::vector<int> selectedIndices(target.size(), -1);
  std::vector<bool> used(dataset.size(), false);

  for (int i = 0; i < target.size(); i++) {
    int bestIndex = -1;
    int bestDist = INT_MAX;

    for (int j = 0; j < dataset.size(); j++) {
      if (used[j])
        continue;
      int dist = (target[i] - dataset[j]) * (target[i] - dataset[j]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIndex = j;
      }
    }

    if (bestIndex != -1) {
      used[bestIndex] = true;
      selectedIndices[i] = bestIndex;
    }
  }

  return selectedIndices;
}

struct Zone {
  int index;
  int difficulty;
};

std::vector<int>
ImageOrdering::orderPriority(const std::vector<unsigned char> &target,
                             const std::vector<unsigned char> &dataset) {

  std::vector<int> selectedIndices(target.size(), -1);
  std::vector<bool> used(dataset.size(), false);
  std::vector<Zone> zones(target.size());

  int threshold = 100;

  // Compute difficulty for each zone
  for (int i = 0; i < target.size(); i++) {
    zones[i].index = i;
    int count = 0;
    for (int j = 0; j < dataset.size(); j++)
      if ((target[i] - dataset[j]) * (target[i] - dataset[j]) < threshold)
        count++;
    zones[i].difficulty = count;
  }

  // Sort zones by difficulty ascending
  std::sort(zones.begin(), zones.end(), [](const Zone &a, const Zone &b) {
    return a.difficulty < b.difficulty;
  });

  // Assign images
  for (int k = 0; k < target.size(); k++) {
    int zoneIdx = zones[k].index;
    unsigned char t = target[zoneIdx];

    int bestIndex = -1;
    int bestDist = INT_MAX;

    for (int j = 0; j < dataset.size(); j++) {
      if (used[j])
        continue;
      int dist = (t - dataset[j]) * (t - dataset[j]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIndex = j;
      }
    }

    if (bestIndex != -1) {
      used[bestIndex] = true;
      selectedIndices[zoneIdx] = bestIndex;
    }
  }

  // Optional improvement by swapping
  bool improved = true;
  int N = selectedIndices.size();
  while (improved) {
    improved = false;
    for (int i = 0; i < N; i++) {
      for (int j = i + 1; j < N; j++) {
        int imgA = selectedIndices[i];
        int imgB = selectedIndices[j];

        int oldCost =
            (target[i] - dataset[imgA]) * (target[i] - dataset[imgA]) +
            (target[j] - dataset[imgB]) * (target[j] - dataset[imgB]);

        int newCost =
            (target[i] - dataset[imgB]) * (target[i] - dataset[imgB]) +
            (target[j] - dataset[imgA]) * (target[j] - dataset[imgA]);

        if (newCost < oldCost) {
          std::swap(selectedIndices[i], selectedIndices[j]);
          improved = true;
        }
      }
    }
  }

  return selectedIndices;
}