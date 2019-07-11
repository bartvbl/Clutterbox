#pragma once

#include <string>
#include <vector>

void runClutterBoxExperiment(
        std::string objectDirectory,
        std::vector<int> objectCountList,
        float boxSize,
        float spinImageWidth,
        float spinImageSupportAngleDegrees,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        size_t overrideSeed = 0);