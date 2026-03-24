#pragma once
#include <vector>

class ImageOrdering {
public:
    static std::vector<int> orderAllowRepeats(const std::vector<unsigned char>& target,
                                              const std::vector<unsigned char>& dataset);
    static std::vector<int> orderUnique(const std::vector<unsigned char>& target,
                                        const std::vector<unsigned char>& dataset);
    static std::vector<int> orderPriority(const std::vector<unsigned char>& target,
                                          const std::vector<unsigned char>& dataset);
};