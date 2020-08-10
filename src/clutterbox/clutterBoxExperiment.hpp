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
        std::vector<Clutterbox::Method*> descriptorsToEvaluate,
        std::vector<int> objectCountList,
        int overrideObjectCount,
        float boxSize,
        float supportRadius,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        bool dumpSceneOBJFiles,
        std::string sceneOBJFileDumpDir,
        bool enableMatchVisualisation,
        std::string matchVisualisationOutputDir,
        std::vector<std::string> matchVisualisationDescriptorList,
        unsigned int matchVisualisationThreshold,
        GPUMetaData gpuMetaData,
        size_t overrideSeed = 0);
