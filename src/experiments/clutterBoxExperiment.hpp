#pragma once

#include <string>
#include <vector>

struct GPUMetaData {
    std::string name;
    int clockRate;
    size_t memorySizeMB;
};

void runClutterBoxExperiment(
        std::string objectDirectory,
        std::vector<std::string> descriptorList,
        std::vector<int> objectCountList,
        int overrideObjectCount,
        float boxSize,
        float spinImageWidth,
        float spinImageSupportAngleDegrees,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        GPUMetaData gpuMetaData,
        size_t overrideSeed = 0);