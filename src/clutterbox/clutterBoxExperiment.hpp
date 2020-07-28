#pragma once

#include <string>
#include <vector>
#include <clutterbox/methods/types/ClutterboxMethod.h>

struct GPUMetaData {
    std::string name;
    int clockRate;
    size_t memorySizeMB;
};

void runClutterBoxExperiment(
        std::string objectDirectory,
        std::vector<ClutterboxMethod*> descriptorList,
        std::vector<int> objectCountList,
        int overrideObjectCount,
        float boxSize,
        float pointDensityRadius3dsc,
        float minSupportRadius3dsc,
        float supportRadius,
        float spinImageSupportAngleDegrees,
        unsigned int fpfhBinCount,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        bool dumpSceneOBJFiles,
        std::string sceneOBJFileDumpDir,
        bool enableMatchVisualisation,
        std::string matchVisualisationOutputDir,
        std::vector<std::string> matchVisualisationDescriptorList,
        GPUMetaData gpuMetaData,
        size_t overrideSeed = 0);