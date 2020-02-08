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
        float pointDensityRadius3dsc,
        float minSupportRadius3dsc,
        float supportRadius,
        float spinImageSupportAngleDegrees,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        bool dumpSceneOBJFiles,
        bool enableMatchVisualisation,
        std::string sceneOBJFileDumpDir,
        GPUMetaData gpuMetaData,
        size_t overrideSeed = 0);