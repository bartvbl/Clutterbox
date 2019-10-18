#pragma once

#include "types/ExperimentSettings.h"
#include "../../../../libShapeSearch/lib/arrrgh/arrrgh.hpp"


ExperimentSettings
generateRandomExperimentSettings(
        std::string objectDirectory,
        std::vector<std::string> descriptorsToGenerateList,
        std::vector<int> objectCountList,
        int overrideObjectCount,
        float boxSize,
        float spinImageWidth,
        float spinImageSupportAngle,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        size_t randomSeed);
