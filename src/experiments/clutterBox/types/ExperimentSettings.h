#pragma once
#include <vector>
#include <string>
#include <glm/vec3.hpp>

struct ExperimentSettings {
    std::string objectDirectory;
    std::string sourceFileDirectory;
    std::string outputDirectory;
    std::vector<std::string> descriptorsToGenerateList;
    size_t seed;
    std::vector<int> objectCountList;
    int overrideObjectCount;
    float boxSize;
    float spinImageWidth;
    float spinImageSupportAngleDegrees;
    std::vector<size_t> spinImageSampleCounts;
    std::vector<std::string> chosenFiles;
    std::vector<glm::vec3> rotations;
    std::vector<glm::vec3> translations;
    unsigned long spinImageReferenceSamplingSeed;
    std::vector<unsigned long> spinImageSampleSamplingSeeds;
    bool dumpRawSearchResults;
    std::vector<std::string> descriptorList;
};