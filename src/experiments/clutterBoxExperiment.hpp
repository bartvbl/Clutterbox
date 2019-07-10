#pragma once

#include <string>

void runClutterBoxExperiment(
        std::string objectDirectory,
        unsigned int sampleSetSize,
        float boxSize,
        float spinImageWidth,
        float spinImageSupportAngleDegrees,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        size_t overrideSeed = 0);