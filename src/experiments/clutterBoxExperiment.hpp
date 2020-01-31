#pragma once

#include <string>
#include <vector>

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
        std::string sceneOBJFileDumpDir,
        size_t overrideSeed = 0);