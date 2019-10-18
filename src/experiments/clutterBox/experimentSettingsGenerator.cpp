#include "experimentSettingsGenerator.h"


#include <random>
#include <iostream>
#include <algorithm>
#include <spinImage/utilities/OBJLoader.h>
#include <experimentUtilities/listDir.h>
#include <utilities/stringUtils.h>

std::vector<std::string> generateRandomFileList(const std::string &objectDirectory, unsigned int sampleSetSize,
                                                std::default_random_engine &generator) {

    std::vector<std::string> filePaths(sampleSetSize);

    std::cout << "\tListing object directory..";
    std::vector<std::string> fileList = listDir(objectDirectory);
    std::cout << " (found " << fileList.size() << " files)" << std::endl;

    // Sort the file list to avoid the effects of operating systems ordering files inconsistently.
    std::sort(fileList.begin(), fileList.end());

    std::shuffle(std::begin(fileList), std::end(fileList), generator);

    for (unsigned int i = 0; i < sampleSetSize; i++) {
        filePaths[i] = objectDirectory + (endsWith(objectDirectory, "/") ? "" : "/") + fileList.at(i);
        std::cout << "\t\tSample " << i << ": " << filePaths.at(i) << std::endl;
    }

    return filePaths;
}

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
        size_t randomSeed) {
    // Forcing the random number generator to be consistent across all platforms
    // This is the same generator as std::default_random_engine on my system.
    std::minstd_rand0 generator{randomSeed};
    std::cout << "Random seed: " << randomSeed << std::endl;

    ExperimentSettings settings;

    int sampleObjectCount = *std::max_element(objectCountList.begin(), objectCountList.end());
    int originalObjectCount = sampleObjectCount;

    if(overrideObjectCount != -1) {
        std::cout << "Using overridden object count: " << overrideObjectCount << std::endl;
        sampleObjectCount = overrideObjectCount;
    }

    std::vector<std::string> chosenFiles = generateRandomFileList(objectDirectory, sampleObjectCount, generator);

    // Technically not needed, because generateRandomFileList() does this to some extent already
    // but kept in for consistency with existing results
    std::shuffle(std::begin(chosenFiles), std::end(chosenFiles), generator);

    // This represents an random number generation for the spin image seed selection
    settings.spinImageReferenceSamplingSeed = generator();

    std::uniform_real_distribution<float> distribution(0, 1);

    std::vector<glm::vec3> rotations(sampleObjectCount);
    std::vector<glm::vec3> translations(sampleObjectCount);

    for(unsigned int i = 0; i < sampleObjectCount; i++) {
        float yaw = float(distribution(generator) * 2.0 * M_PI);
        float pitch = float((distribution(generator) - 0.5) * M_PI);
        float roll = float(distribution(generator) * 2.0 * M_PI);

        float distanceX = boxSize * distribution(generator);
        float distanceY = boxSize * distribution(generator);
        float distanceZ = boxSize * distribution(generator);

        rotations.at(i) = glm::vec3(yaw, pitch, roll);
        translations.at(i) = glm::vec3(distanceX, distanceY, distanceZ);

        std::cout << "\tRotation: (" << yaw << ", " << pitch << ", "<< roll << "), Translation: (" << distanceX << ", "<< distanceY << ", "<< distanceZ << ")" << std::endl;
    }

    std::vector<HostMesh> sampleMeshes(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        sampleMeshes.at(i) = SpinImage::utilities::loadOBJ(chosenFiles.at(i), true);
    }

    // This picks random seeds for the spin image sampling
    settings.spinImageSampleSamplingSeeds.resize(objectCountList.size());
    for(int i = 0; i < objectCountList.size(); i++) {
        settings.spinImageSampleSamplingSeeds.at(i) = generator();
    }

    settings.objectDirectory = objectDirectory;
    settings.sourceFileDirectory = ;
    settings.outputDirectory;
    settings.descriptorsToGenerateList;
    settings.seed;
    settings.objectCountList;
    settings.overrideObjectCount;
    settings.boxSize;
    settings.spinImageWidth;
    settings.spinImageSupportAngleDegrees;
    settings.spinImageSampleCounts;
    settings.chosenFiles;
    settings.rotations;
    settings.translations;
    settings.spinImageReferenceSamplingSeed;
    settings.spinImageSampleSamplingSeeds;
    settings.dumpRawSearchResults;
    settings.descriptorList;

    return settings;
}