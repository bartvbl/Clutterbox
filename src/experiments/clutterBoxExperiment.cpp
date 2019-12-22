#include "clutterBoxExperiment.hpp"

#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <algorithm>

#include <utilities/stringUtils.h>
#include <spinImage/utilities/modelScaler.h>
#include <utilities/Histogram.h>

#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/radialIntersectionCountImageGenerator.cuh>
#include <spinImage/gpu/radialIntersectionCountImageSearcher.cuh>
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>
#include <spinImage/gpu/quickIntersectionCountImageSearcher.cuh>
#include <spinImage/gpu/spinImageGenerator.cuh>
#include <spinImage/gpu/spinImageSearcher.cuh>
#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/utilities/copy/deviceDescriptorsToHost.h>
#include <spinImage/utilities/dumpers/spinImageDumper.h>
#include <spinImage/utilities/dumpers/searchResultDumper.h>
#include <spinImage/utilities/duplicateRemoval.cuh>
#include <spinImage/utilities/modelScaler.h>

#include <experiments/clutterBox/clutterBoxUtilities.h>
#include <fstream>
#include <glm/vec3.hpp>
#include <map>
#include <sstream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <json.hpp>
#include <tsl/ordered_map.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

#include "clutterBox/clutterBoxKernels.cuh"

#include "experimentUtilities/listDir.h"
#include "nvidia/helper_cuda.h"


// TODO list:
// - The measure's independent variable should not be number of objects, but rather the number of triangles in the scene
// - How do I manage samples in the scene for spin images? Certain number of samples per triangle?
// - What is the effect of different spin image sizes?
// - In order to limit VRAM usage, as well as get a better signal to noise ratio (due to aliasing) on the images, we should only use models with less than a certain number of vertices.
// -



Histogram computeSearchResultHistogram(size_t vertexCount, const SpinImage::array<unsigned int> &searchResults);

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

void dumpResultsFile(std::string outputFile, std::vector<std::string> descriptorList, size_t seed,
                     std::vector<Histogram> RICIHistograms, std::vector<Histogram> QUICCIHistograms,
                     std::vector<Histogram> SIHistograms,
                     const std::string &sourceFileDirectory, std::vector<int> objectCountList, int overrideObjectCount,
                     float boxSize, float spinImageWidth, size_t assertionRandomToken,
                     std::vector<SpinImage::debug::RICIRunInfo> RICIRuns,
                     std::vector<SpinImage::debug::QUICCIRunInfo> QUICCIRuns,
                     std::vector<SpinImage::debug::SIRunInfo> SIRuns,
                     std::vector<SpinImage::debug::RICISearchRunInfo> RICISearchRuns,
                     std::vector<SpinImage::debug::QUICCISearchRunInfo> QUICCISearchRuns,
                     std::vector<SpinImage::debug::SISearchRunInfo> SISearchRuns, float spinImageSupportAngleDegrees,
                     std::vector<size_t> uniqueVertexCounts, std::vector<size_t> spinImageSampleCounts,
                     GPUMetaData gpuMetaData) {
    std::cout << std::endl << "Dumping results file.." << std::endl;

    std::default_random_engine generator{seed};

    int sampleObjectCount = *std::max_element(objectCountList.begin(), objectCountList.end());
    int originalObjectCount = sampleObjectCount;

    if(overrideObjectCount != -1) {
        std::cout << "Using overridden object count: " << overrideObjectCount << std::endl;
        sampleObjectCount = overrideObjectCount;
    }

    std::vector<std::string> chosenFiles = generateRandomFileList(sourceFileDirectory, sampleObjectCount, generator);

    std::shuffle(std::begin(chosenFiles), std::end(chosenFiles), generator);

    // This represents an random number generation for the spin image seed selection
    generator();

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

    std::vector<SpinImage::cpu::Mesh> sampleMeshes(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        sampleMeshes.at(i) = SpinImage::utilities::loadOBJ(chosenFiles.at(i), true);
    }

    // This represents random number generations for the spin image seed selections during the experiment
    for(int i = 0; i < objectCountList.size(); i++) {
        generator();
    }

    size_t finalCheckToken = generator();
    if(finalCheckToken != assertionRandomToken) {
        std::cerr << "ERROR: the verification token generated by the metadata dump function was different than the one used to generate the program output. This means that due to changes the metadata computed by the dump function is likely wrong and should be corrected." << std::endl;
        std::cerr << "Expected: " << finalCheckToken << ", got: " << assertionRandomToken << std::endl;
    }

    bool riciDescriptorActive = false;
    bool quicciDescriptorActive = false;
    bool siDescriptorActive = false;

    for(const auto& descriptor : descriptorList) {
        if(descriptor == "rici") {
            riciDescriptorActive = true;
        } else if(descriptor == "quicci") {
            quicciDescriptorActive = true;
        } else if(descriptor == "si") {
            siDescriptorActive = true;
        }
    }

    json outJson;

    outJson["version"] = "v10";
    outJson["seed"] = seed;
    outJson["descriptors"] = descriptorList;
    outJson["sampleSetSize"] = sampleObjectCount;
    outJson["sampleObjectCounts"] = objectCountList;
    outJson["overrideObjectCount"] = overrideObjectCount;
    outJson["uniqueVertexCounts"] = uniqueVertexCounts;
    outJson["imageCounts"] = uniqueVertexCounts;
    outJson["boxSize"] = boxSize;
    outJson["spinImageWidth"] = spinImageWidth;
    outJson["spinImageWidthPixels"] = spinImageWidthPixels;
    outJson["spinImageSupportAngle"] = spinImageSupportAngleDegrees;
    outJson["spinImageSampleCounts"] = spinImageSampleCounts;
    outJson["searchResultCount"] = SEARCH_RESULT_COUNT;
    outJson["inputFiles"] = chosenFiles;
    outJson["riciEarlyExitEnabled"] = ENABLE_RICI_COMPARISON_EARLY_EXIT;
    outJson["riciSharedMemoryImageEnabled"] = ENABLE_SHARED_MEMORY_IMAGE;
    outJson["gpuInfo"] = {};
    outJson["gpuInfo"]["name"] = gpuMetaData.name;
    outJson["gpuInfo"]["clockrate"] = gpuMetaData.clockRate;
    outJson["gpuInfo"]["memoryCapacity"] = gpuMetaData.memorySizeMB;
    outJson["vertexCounts"] = {};
    for (auto &sampleMesh : sampleMeshes) {
        outJson["vertexCounts"].push_back(sampleMesh.vertexCount);
    }
    outJson["rotations"] = {};
    for (auto &rotation : rotations) {
        outJson["rotations"].push_back({rotation.x, rotation.y, rotation.z});
    }
    outJson["translations"] = {};
    for (auto &translation : translations) {
        outJson["translations"].push_back({translation.x, translation.y, translation.z});
    }

    if(riciDescriptorActive) {
        outJson["runtimes"]["RICIReferenceGeneration"]["total"] = RICIRuns.at(0).totalExecutionTimeSeconds;
        outJson["runtimes"]["RICIReferenceGeneration"]["meshScale"] = RICIRuns.at(0).meshScaleTimeSeconds;
        outJson["runtimes"]["RICIReferenceGeneration"]["redistribution"] = RICIRuns.at(0).redistributionTimeSeconds;
        outJson["runtimes"]["RICIReferenceGeneration"]["generation"] = RICIRuns.at(0).generationTimeSeconds;

        outJson["runtimes"]["RICISampleGeneration"]["total"] = {};
        outJson["runtimes"]["RICISampleGeneration"]["meshScale"] = {};
        outJson["runtimes"]["RICISampleGeneration"]["redistribution"] = {};
        outJson["runtimes"]["RICISampleGeneration"]["generation"] = {};
        for(unsigned int i = 1; i < RICIRuns.size(); i++) {
            outJson["runtimes"]["RICISampleGeneration"]["total"].push_back(RICIRuns.at(i).totalExecutionTimeSeconds);
            outJson["runtimes"]["RICISampleGeneration"]["meshScale"].push_back(RICIRuns.at(i).meshScaleTimeSeconds);
            outJson["runtimes"]["RICISampleGeneration"]["redistribution"].push_back(RICIRuns.at(i).redistributionTimeSeconds);
            outJson["runtimes"]["RICISampleGeneration"]["generation"].push_back(RICIRuns.at(i).generationTimeSeconds);
        }
    }

    if(quicciDescriptorActive) {
        outJson["runtimes"]["QUICCIReferenceGeneration"]["total"] = QUICCIRuns.at(0).totalExecutionTimeSeconds;
        outJson["runtimes"]["QUICCIReferenceGeneration"]["generation"] = QUICCIRuns.at(0).generationTimeSeconds;

        outJson["runtimes"]["QUICCISampleGeneration"]["total"] = {};
        outJson["runtimes"]["QUICCISampleGeneration"]["generation"] = {};
        for(unsigned int i = 1; i < RICIRuns.size(); i++) {
            outJson["runtimes"]["QUICCISampleGeneration"]["total"].push_back(QUICCIRuns.at(i).totalExecutionTimeSeconds);
            outJson["runtimes"]["QUICCISampleGeneration"]["generation"].push_back(QUICCIRuns.at(i).generationTimeSeconds);
        }
    }

    if(siDescriptorActive) {
        outJson["runtimes"]["SIReferenceGeneration"]["total"] = SIRuns.at(0).totalExecutionTimeSeconds;
        outJson["runtimes"]["SIReferenceGeneration"]["initialisation"] = SIRuns.at(0).initialisationTimeSeconds;
        outJson["runtimes"]["SIReferenceGeneration"]["sampling"] = SIRuns.at(0).meshSamplingTimeSeconds;
        outJson["runtimes"]["SIReferenceGeneration"]["generation"] = SIRuns.at(0).generationTimeSeconds;

        outJson["runtimes"]["SISampleGeneration"]["total"] = {};
        outJson["runtimes"]["SISampleGeneration"]["initialisation"] = {};
        outJson["runtimes"]["SISampleGeneration"]["sampling"] = {};
        outJson["runtimes"]["SISampleGeneration"]["generation"] = {};
        for(unsigned int i = 1; i < SIRuns.size(); i++) {
            outJson["runtimes"]["SISampleGeneration"]["total"].push_back(SIRuns.at(i).totalExecutionTimeSeconds);
            outJson["runtimes"]["SISampleGeneration"]["initialisation"].push_back(SIRuns.at(i).initialisationTimeSeconds);
            outJson["runtimes"]["SISampleGeneration"]["sampling"].push_back(SIRuns.at(i).meshSamplingTimeSeconds);
            outJson["runtimes"]["SISampleGeneration"]["generation"].push_back(SIRuns.at(i).generationTimeSeconds);
        }
    }

    if(riciDescriptorActive) {
        outJson["runtimes"]["RICISearch"]["total"] = {};
        outJson["runtimes"]["RICISearch"]["search"] = {};
        for (auto &RICISearchRun : RICISearchRuns) {
            outJson["runtimes"]["RICISearch"]["total"].push_back(RICISearchRun.totalExecutionTimeSeconds);
            outJson["runtimes"]["RICISearch"]["search"].push_back(RICISearchRun.searchExecutionTimeSeconds);
        }
    }

    if(quicciDescriptorActive) {
        outJson["runtimes"]["QUICCISearch"]["total"] = {};
        outJson["runtimes"]["QUICCISearch"]["search"] = {};
        for (auto &QUICCISearchRun : QUICCISearchRuns) {
            outJson["runtimes"]["QUICCISearch"]["total"].push_back(QUICCISearchRun.totalExecutionTimeSeconds);
            outJson["runtimes"]["QUICCISearch"]["search"].push_back(QUICCISearchRun.searchExecutionTimeSeconds);
        }
    }

    if(siDescriptorActive) {
        outJson["runtimes"]["SISearch"]["total"] = {};
        outJson["runtimes"]["SISearch"]["averaging"] = {};
        outJson["runtimes"]["SISearch"]["search"] = {};
        for (auto &SISearchRun : SISearchRuns) {
            outJson["runtimes"]["SISearch"]["total"].push_back(SISearchRun.totalExecutionTimeSeconds);
            outJson["runtimes"]["SISearch"]["averaging"].push_back(SISearchRun.averagingExecutionTimeSeconds);
            outJson["runtimes"]["SISearch"]["search"].push_back(SISearchRun.searchExecutionTimeSeconds);
        }
    }

    if(riciDescriptorActive) {
        outJson["RICIhistograms"] = {};
        for(unsigned int i = 0; i < objectCountList.size(); i++) {
            std::map<unsigned int, size_t> riciMap = RICIHistograms.at(i).getMap();
            std::vector<unsigned int> keys;
            for (auto &content : riciMap) {
                keys.push_back(content.first);
            }
            std::sort(keys.begin(), keys.end());

            for(auto &key : keys) {
                outJson["RICIhistograms"][std::to_string(objectCountList.at(i)) + " objects"][std::to_string(key)] = riciMap[key];
            }
        }
    }

    if(quicciDescriptorActive) {
        outJson["QUICCIhistograms"] = {};
        for(unsigned int i = 0; i < objectCountList.size(); i++) {
            std::map<unsigned int, size_t> quicciMap = QUICCIHistograms.at(i).getMap();
            std::vector<unsigned int> keys;
            for (auto &content : quicciMap) {
                keys.push_back(content.first);
            }
            std::sort(keys.begin(), keys.end());

            for(auto &key : keys) {
                outJson["QUICCIhistograms"][std::to_string(objectCountList.at(i)) + " objects"][std::to_string(key)] = quicciMap[key];
            }
        }
    }

    if(siDescriptorActive) {
        outJson["SIhistograms"] = {};
        for(unsigned int i = 0; i < objectCountList.size(); i++) {
            std::map<unsigned int, size_t> siMap = SIHistograms.at(i).getMap();
            std::vector<unsigned int> keys;
            for (auto &content : siMap) {
                keys.push_back(content.first);
            }
            std::sort(keys.begin(), keys.end());

            for(auto &key : keys) {
                outJson["SIhistograms"][std::to_string(objectCountList.at(i)) + " objects"][std::to_string(key)] = siMap[key];
            }
        }
    }

    std::ofstream outFile(outputFile);
    outFile << outJson.dump(4);
    outFile.close();

    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        SpinImage::cpu::freeMesh(sampleMeshes.at(i));
    }
}

void dumpRawSearchResultFile(std::string outputFile, std::vector<std::string> descriptorList,
                             std::vector<int> objectCountList,
                             std::vector<SpinImage::array<unsigned int>> rawRICISearchResults,
                             std::vector<SpinImage::array<unsigned int>> rawQUICCISearchResults,
                             std::vector<SpinImage::array<unsigned int>> rawSISearchResults,
                             size_t seed) {

    json outJson;

    outJson["version"] = "rawfile_v3";
    outJson["sampleObjectCounts"] = objectCountList;
    outJson["seed"] = seed;

    bool riciDescriptorActive = false;
    bool quicciDescriptorActive = false;
    bool siDescriptorActive = false;

    for(const auto& descriptor : descriptorList) {
        if(descriptor == "rici") {
            riciDescriptorActive = true;
        } else if(descriptor == "quicci") {
            quicciDescriptorActive = true;
        } else if(descriptor == "si") {
            siDescriptorActive = true;
        }
    }

    // RICI block
    if(riciDescriptorActive) {
        outJson["RICI"] = {};
        for(int i = 0; i < rawRICISearchResults.size(); i++) {
            std::string indexString = std::to_string(objectCountList.at(i));
            outJson["RICI"][indexString] = {};
            for(int j = 0; j < rawRICISearchResults.at(i).length; j++) {
                outJson["RICI"][indexString].push_back(rawRICISearchResults.at(i).content[j]);
            }
        }
    }

    // QUICCI block
    if(quicciDescriptorActive) {
        outJson["QUICCI"] = {};
        for(int i = 0; i < rawQUICCISearchResults.size(); i++) {
            std::string indexString = std::to_string(objectCountList.at(i));
            outJson["QUICCI"][indexString] = {};
            for(int j = 0; j < rawQUICCISearchResults.at(i).length; j++) {
                outJson["QUICCI"][indexString].push_back(rawQUICCISearchResults.at(i).content[j]);
            }
        }
    }

    if(siDescriptorActive) {
        // SI block
        outJson["SI"] = {};
        for(int i = 0; i < rawSISearchResults.size(); i++) {
            std::string indexString = std::to_string(objectCountList.at(i));
            outJson["SI"][indexString] = {};
            for(int j = 0; j < rawSISearchResults.at(i).length; j++) {
                outJson["SI"][indexString].push_back(rawSISearchResults.at(i).content[j]);
            }
        }
    }

    std::ofstream outFile(outputFile);
    outFile << outJson.dump(4);
    outFile.close();
}







const inline size_t computeSpinImageSampleCount(size_t &vertexCount) {
    return std::max((size_t)1000000, (size_t) (30 * vertexCount)); 
}

void dumpSpinImages(std::string filename, SpinImage::array<spinImagePixelType> device_descriptors) {
    size_t arrayLength = std::min(device_descriptors.length, (size_t)2500);
    SpinImage::array<float> hostDescriptors = SpinImage::copy::spinImageDescriptorsToHost(device_descriptors, arrayLength);
    hostDescriptors.length = arrayLength;
    SpinImage::dump::descriptors(hostDescriptors, filename, true, 50);
    delete[] hostDescriptors.content;
}

void dumpRadialIntersectionCountImages(std::string filename, SpinImage::array<radialIntersectionCountImagePixelType> device_descriptors) {
    size_t arrayLength = std::min(device_descriptors.length, (size_t)2500);
    SpinImage::array<radialIntersectionCountImagePixelType > hostDescriptors = SpinImage::copy::RICIDescriptorsToHost(device_descriptors, std::min(device_descriptors.length, (size_t)2500));
    hostDescriptors.length = arrayLength;
    SpinImage::dump::descriptors(hostDescriptors, filename, true, 50);
    delete[] hostDescriptors.content;
}

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
        size_t overrideSeed) {

    // Determine which algorithms to enable
    bool riciDescriptorActive = false;
    bool siDescriptorActive = false;
    bool quicciDescriptorActive = false;

    std::cout << "Running clutterbox experiment for the following descriptors: ";

    for(const auto& descriptor : descriptorList) {
        if(descriptor == "rici") {
            riciDescriptorActive = true;
            std::cout << (quicciDescriptorActive || siDescriptorActive ? ", " : "") << "Radial Intersection Count Image";
        } else if(descriptor == "si") {
            siDescriptorActive = true;
            std::cout << (quicciDescriptorActive || riciDescriptorActive ? ", " : "") << "Spin Image";
        } else if(descriptor == "quicci") {
            quicciDescriptorActive = true;
            std::cout << (riciDescriptorActive || siDescriptorActive ? ", " : "") << "Quick Intersection Count Change Image";
        }
    }
    std::cout << std::endl;


    // --- Overview ---
	//
	// 1 Search SHREC directory for files
	// 2 Make a sample set of n sample objects
	// 3 Load the models in the sample set
	// 4 Scale all models to fit in a 1x1x1 sphere
	// 5 Compute radial intersection count and spin images for all models in the sample set
	// 6 Create a box of SxSxS units
	// 7 for all combinations (non-reused) models:
	//    7.1 Place each mesh in the box, retrying if it collides with another mesh
	//    7.2 For all meshes in the box, compute spin images for all vertices
	//    7.3 Compare the generated images to the "clutter-free" variants
	//    7.4 Dump the distances between images

    std::vector<Histogram> RICIHistograms;
    std::vector<Histogram> spinImageHistograms;
    std::vector<Histogram> QUICCIHistograms;

    std::vector<SpinImage::debug::RICIRunInfo> RICIRuns;
    std::vector<SpinImage::debug::SIRunInfo> SIRuns;
    std::vector<SpinImage::debug::QUICCIRunInfo> QUICCIRuns;

    std::vector<SpinImage::debug::SISearchRunInfo> SISearchRuns;
    std::vector<SpinImage::debug::RICISearchRunInfo> RICISearchRuns;
    std::vector<SpinImage::debug::QUICCISearchRunInfo> QUICCISearchRuns;

    // The number of sample objects that need to be loaded depends on the largest number of objects required in the list
    int sampleObjectCount = *std::max_element(objectCountList.begin(), objectCountList.end());

    if(overrideObjectCount != -1) {
        if(overrideObjectCount < sampleObjectCount) {
            std::cout << "ERROR: override object count is lower than highest count specified in object count list!" << std::endl;
            return;
        }

        std::cout << "Using overridden object count: " << overrideObjectCount << std::endl;
        sampleObjectCount = overrideObjectCount;
    }

    // 1 Seeding the random number generator
    std::random_device rd;
    size_t randomSeed = overrideSeed != 0 ? overrideSeed : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::minstd_rand0 generator{randomSeed};

    std::cout << std::endl << "Running experiment initialisation sequence.." << std::endl;

    // 2 Search SHREC directory for files
    // 3 Make a sample set of n sample objects
    std::vector<std::string> filePaths = generateRandomFileList(objectDirectory, sampleObjectCount, generator);

    // 4 Load the models in the sample set
    std::cout << "\tLoading sample models.." << std::endl;
    std::vector<SpinImage::cpu::Mesh> sampleMeshes(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        sampleMeshes.at(i) = SpinImage::utilities::loadOBJ(filePaths.at(i), true);
        std::cout << "\t\tMesh " << i << ": " << sampleMeshes.at(i).vertexCount << " vertices" << std::endl;
    }

    // 5 Scale all models to fit in a 1x1x1 sphere
    std::cout << "\tScaling meshes.." << std::endl;
    std::vector<SpinImage::cpu::Mesh> scaledMeshes(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        scaledMeshes.at(i) = SpinImage::utilities::fitMeshInsideSphereOfRadius(sampleMeshes.at(i), 1);
        SpinImage::cpu::freeMesh(sampleMeshes.at(i));
    }

    // 6 Copy meshes to GPU
    std::cout << "\tCopying meshes to device.." << std::endl;
    std::vector<SpinImage::gpu::Mesh> scaledMeshesOnGPU(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        scaledMeshesOnGPU.at(i) = SpinImage::copy::hostMeshToDevice(scaledMeshes.at(i));
    }

    // 7 Shuffle the list. First mesh is now our "reference".
    std::cout << "\tShuffling sample object list.." << std::endl;
    std::shuffle(std::begin(scaledMeshesOnGPU), std::end(scaledMeshesOnGPU), generator);

    // 8 Remove duplicate vertices
    std::cout << "\tRemoving duplicate vertices.." << std::endl;
    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> spinOrigins_reference = SpinImage::utilities::computeUniqueVertices(scaledMeshesOnGPU.at(0));
    size_t referenceImageCount = spinOrigins_reference.length;
    std::cout << "\t\tReduced " << scaledMeshesOnGPU.at(0).vertexCount << " vertices to " << referenceImageCount << "." << std::endl;

    size_t spinImageSampleCount = computeSpinImageSampleCount(scaledMeshesOnGPU.at(0).vertexCount);
    std::cout << "\tUsing sample count: " << spinImageSampleCount << std::endl;

    // 9 Compute spin image for reference model
    SpinImage::array<radialIntersectionCountImagePixelType> device_referenceRICIImages;
    SpinImage::array<spinImagePixelType> device_referenceSpinImages;
    SpinImage::gpu::QUICCIImages device_referenceQuiccImages;

    if(riciDescriptorActive) {
        std::cout << "\tGenerating reference RICI images.. (" << referenceImageCount << " images)" << std::endl;
        SpinImage::debug::RICIRunInfo riciReferenceRunInfo;
        device_referenceRICIImages = SpinImage::gpu::generateRadialIntersectionCountImages(
                scaledMeshesOnGPU.at(0),
                spinOrigins_reference,
                spinImageWidth,
                &riciReferenceRunInfo);

        RICIRuns.push_back(riciReferenceRunInfo);
        std::cout << "\t\tExecution time: " << riciReferenceRunInfo.generationTimeSeconds << std::endl;

        // Requires results from RICI descriptors, therefore put inside here
        if(quicciDescriptorActive) {
            std::cout << "\tGenerating QUICCI images.." << std::endl;

            SpinImage::debug::QUICCIRunInfo quicciReferenceRunInfo;
            device_referenceQuiccImages = SpinImage::gpu::generateQUICCImages(
                    device_referenceRICIImages,
                    &quicciReferenceRunInfo
            );

            QUICCIRuns.push_back(quicciReferenceRunInfo);
            std::cout << "\t\tExecution time: " << quicciReferenceRunInfo.generationTimeSeconds << std::endl;
        }
    }

    if(siDescriptorActive) {
        std::cout << "\tGenerating reference spin images.." << std::endl;
        SpinImage::debug::SIRunInfo siReferenceRunInfo;
        device_referenceSpinImages = SpinImage::gpu::generateSpinImages(
                scaledMeshesOnGPU.at(0),
                spinOrigins_reference,
                spinImageWidth,
                spinImageSampleCount,
                spinImageSupportAngleDegrees,
                generator(),
                &siReferenceRunInfo);

        SIRuns.push_back(siReferenceRunInfo);
        std::cout << "\t\tExecution time: " << siReferenceRunInfo.generationTimeSeconds << std::endl;
    } else {
        // This keeps the random number generator in a constant state
        // Generating spin images causes a single random number to be generated.
        generator();
    }

    checkCudaErrors(cudaFree(spinOrigins_reference.content));

    // 10 Combine meshes into one larger scene
    SpinImage::gpu::Mesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

    // 11 Compute unique vertex mapping
    std::vector<size_t> uniqueVertexCounts;
    size_t totalUniqueVertexCount;
    SpinImage::array<signed long long> device_indexMapping = SpinImage::utilities::computeUniqueIndexMapping(boxScene, scaledMeshesOnGPU, &uniqueVertexCounts, totalUniqueVertexCount);

    // 12 Randomly transform objects
    std::cout << "\tRandomly transforming input objects.." << std::endl;
    randomlyTransformMeshes(boxScene, boxSize, scaledMeshesOnGPU, generator);

    size_t vertexCount = 0;
    size_t referenceMeshImageCount = spinOrigins_reference.length;

    // 13 Compute corresponding transformed vertex buffer
    //    A mapping is used here because the previously applied transformation can cause non-unique vertices to become
    //    equivalent. It is vital we can rely on a 1:1 mapping existing between vertices.
    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_uniqueSpinOrigins = SpinImage::utilities::applyUniqueMapping(boxScene, device_indexMapping, totalUniqueVertexCount);
    checkCudaErrors(cudaFree(device_indexMapping.content));
    size_t imageCount = 0;

    // 14 Ensure enough memory is available to complete the experiment.
    std::cout << "\tTesting for sufficient memory capacity on GPU.. ";
    int* device_largestNecessaryImageBuffer;
    size_t largestImageBufferSize = totalUniqueVertexCount * spinImageWidthPixels * spinImageWidthPixels * sizeof(int);
    checkCudaErrors(cudaMalloc((void**) &device_largestNecessaryImageBuffer, largestImageBufferSize));
    checkCudaErrors(cudaFree(device_largestNecessaryImageBuffer));
    std::cout << "Success." << std::endl;

    std::vector<SpinImage::array<unsigned int>> rawRICISearchResults;
    std::vector<SpinImage::array<unsigned int>> rawQUICCISearchResults;
    std::vector<SpinImage::array<unsigned int>> rawSISearchResults;
    std::vector<size_t> spinImageSampleCounts;

    int currentObjectListIndex = 0;

    // Generate images for increasingly more complex scenes
    for (int objectCount = 0; objectCount < sampleObjectCount; objectCount++) {
        std::cout << std::endl << "Processing mesh sample " << (objectCount + 1) << "/" << sampleObjectCount << std::endl;
        // Making the generation algorithm believe the scene is smaller than it really is
        // This allows adding objects one by one, without having to copy memory all over the place
        vertexCount += scaledMeshesOnGPU.at(objectCount).vertexCount;
        boxScene.vertexCount = vertexCount;
        imageCount += uniqueVertexCounts.at(objectCount);
        device_uniqueSpinOrigins.length = imageCount;
        std::cout << "\t\tVertex count: " << boxScene.vertexCount << ", Image count: " << imageCount << std::endl;

        // If the object count is not on the list, skip it.
        if(currentObjectListIndex >= objectCountList.size() || (objectCount + 1) != objectCountList.at(currentObjectListIndex)) {
            std::cout << "\tSample count is not on the list. Skipping." << std::endl;
            continue;
        }

        // Marking the current object count as processed
        currentObjectListIndex++;

        // Generating radial intersection count images
        if(riciDescriptorActive) {
            std::cout << "\tGenerating RICI images.. (" << imageCount << " images)" << std::endl;
            SpinImage::debug::RICIRunInfo riciSampleRunInfo;
            SpinImage::array<radialIntersectionCountImagePixelType> device_sampleRICIImages = SpinImage::gpu::generateRadialIntersectionCountImages(
                    boxScene,
                    device_uniqueSpinOrigins,
                    spinImageWidth,
                    &riciSampleRunInfo);
            RICIRuns.push_back(riciSampleRunInfo);
            std::cout << "\t\tTimings: (total " << riciSampleRunInfo.totalExecutionTimeSeconds
                      << ", scaling " << riciSampleRunInfo.meshScaleTimeSeconds
                      << ", redistribution " << riciSampleRunInfo.redistributionTimeSeconds
                      << ", generation " << riciSampleRunInfo.generationTimeSeconds << ")" << std::endl;

            std::cout << "\tSearching in radial intersection count images.." << std::endl;
            SpinImage::debug::RICISearchRunInfo riciSearchRun;
            SpinImage::array<unsigned int> RICIsearchResults = SpinImage::gpu::computeRadialIntersectionCountImageSearchResultRanks(
                    device_referenceRICIImages,
                    referenceMeshImageCount,
                    device_sampleRICIImages,
                    imageCount,
                    &riciSearchRun);
            RICISearchRuns.push_back(riciSearchRun);
            rawRICISearchResults.push_back(RICIsearchResults);
            std::cout << "\t\tTimings: (total " << riciSearchRun.totalExecutionTimeSeconds
                      << ", searching " << riciSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
            Histogram RICIHistogram = computeSearchResultHistogram(referenceMeshImageCount, RICIsearchResults);
            if(!dumpRawSearchResults) {
                delete[] RICIsearchResults.content;
            }

            // Storing results
            RICIHistograms.push_back(RICIHistogram);

            if(quicciDescriptorActive) {
                std::cout << "\tGenerating QUICCI images.. (" << imageCount << " images)" << std::endl;
                SpinImage::debug::QUICCIRunInfo quicciSampleRunInfo;
                SpinImage::gpu::QUICCIImages device_sampleQUICCImages = SpinImage::gpu::generateQUICCImages(
                        device_sampleRICIImages,
                        &quicciSampleRunInfo);
                QUICCIRuns.push_back(quicciSampleRunInfo);
                std::cout << "\t\tTimings: (total " << quicciSampleRunInfo.totalExecutionTimeSeconds
                          << ", generation " << quicciSampleRunInfo.generationTimeSeconds << ")" << std::endl;

                std::cout << "\tSearching in QUICC images.." << std::endl;
                SpinImage::debug::QUICCISearchRunInfo quicciSearchRun;
                SpinImage::array<unsigned int> QUICCIsearchResults = SpinImage::gpu::computeQUICCImageSearchResultRanks(
                        device_referenceQuiccImages,
                        referenceMeshImageCount,
                        device_sampleQUICCImages,
                        imageCount,
                        &quicciSearchRun);
                QUICCISearchRuns.push_back(quicciSearchRun);
                rawQUICCISearchResults.push_back(QUICCIsearchResults);
                std::cout << "\t\tTimings: (total " << quicciSearchRun.totalExecutionTimeSeconds
                          << ", searching " << quicciSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
                Histogram QUICCIHistogram = computeSearchResultHistogram(referenceMeshImageCount, QUICCIsearchResults);
                cudaFree(device_sampleQUICCImages.horizontallyDecreasingImages);
                cudaFree(device_sampleQUICCImages.horizontallyIncreasingImages);
                if(!dumpRawSearchResults) {
                    delete[] QUICCIsearchResults.content;
                }

                // Storing results
                QUICCIHistograms.push_back(QUICCIHistogram);
            }

            // Finally, delete the RICI descriptor images
            cudaFree(device_sampleRICIImages.content);
        }



        // Generating spin images
        if(siDescriptorActive) {
            spinImageSampleCount = computeSpinImageSampleCount(imageCount);
            spinImageSampleCounts.push_back(spinImageSampleCount);
            std::cout << "\tGenerating spin images.. (" << imageCount << " images, " << spinImageSampleCount << " samples)" << std::endl;
            SpinImage::debug::SIRunInfo siSampleRunInfo;
            SpinImage::array<spinImagePixelType> device_sampleSpinImages = SpinImage::gpu::generateSpinImages(
                    boxScene,
                    device_uniqueSpinOrigins,
                    spinImageWidth,
                    spinImageSampleCount,
                    spinImageSupportAngleDegrees,
                    generator(),
                    &siSampleRunInfo);
            SIRuns.push_back(siSampleRunInfo);
            std::cout << "\t\tTimings: (total " << siSampleRunInfo.totalExecutionTimeSeconds
                      << ", initialisation " << siSampleRunInfo.initialisationTimeSeconds
                      << ", sampling " << siSampleRunInfo.meshSamplingTimeSeconds
                      << ", generation " << siSampleRunInfo.generationTimeSeconds << ")" << std::endl;

            std::cout << "\tSearching in spin images.." << std::endl;
            SpinImage::debug::SISearchRunInfo siSearchRun;
            SpinImage::array<unsigned int> SpinImageSearchResults = SpinImage::gpu::computeSpinImageSearchResultRanks(
                    device_referenceSpinImages,
                    referenceMeshImageCount,
                    device_sampleSpinImages,
                    imageCount,
                    &siSearchRun);
            SISearchRuns.push_back(siSearchRun);
            rawSISearchResults.push_back(SpinImageSearchResults);
            std::cout << "\t\tTimings: (total " << siSearchRun.totalExecutionTimeSeconds
                      << ", averaging " << siSearchRun.averagingExecutionTimeSeconds
                      << ", searching " << siSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
            Histogram SIHistogram = computeSearchResultHistogram(referenceMeshImageCount, SpinImageSearchResults);
            cudaFree(device_sampleSpinImages.content);
            if(!dumpRawSearchResults) {
                delete[] SpinImageSearchResults.content;
            }

            // Storing results
            spinImageHistograms.push_back(SIHistogram);
        } else {
            // Keeping the random number generator in sync
            generator();
        }
    }

    SpinImage::gpu::freeMesh(boxScene);
    cudaFree(device_referenceRICIImages.content);
    cudaFree(device_referenceSpinImages.content);
    cudaFree(device_referenceQuiccImages.horizontallyIncreasingImages);
    cudaFree(device_referenceQuiccImages.horizontallyDecreasingImages);
    cudaFree(device_uniqueSpinOrigins.content);

    std::string timestring = getCurrentDateTimeString();

    dumpResultsFile(
            outputDirectory + timestring + "_" + std::to_string(randomSeed) + ".json",
            descriptorList,
            randomSeed,
            RICIHistograms,
            QUICCIHistograms,
            spinImageHistograms,
            objectDirectory,
            objectCountList,
            overrideObjectCount,
            boxSize,
            spinImageWidth,
            generator(),
            RICIRuns,
            QUICCIRuns,
            SIRuns,
            RICISearchRuns,
            QUICCISearchRuns,
            SISearchRuns,
            spinImageSupportAngleDegrees,
            uniqueVertexCounts,
            spinImageSampleCounts,
            gpuMetaData);

    if(dumpRawSearchResults) {
        dumpRawSearchResultFile(
                outputDirectory + "raw/" + timestring + "_" + std::to_string(randomSeed) + ".json",
                descriptorList,
                objectCountList,
                rawRICISearchResults,
                rawQUICCISearchResults,
                rawSISearchResults,
                randomSeed);

        // Cleanup
        // If one of the descriptors is not enabled, this will iterate over an empty vector.
        for(auto results : rawRICISearchResults) {
            delete[] results.content;
        }
        for(auto results : rawQUICCISearchResults) {
            delete[] results.content;
        }
        for(auto results : rawSISearchResults) {
            delete[] results.content;
        }
    }

    for(SpinImage::gpu::Mesh deviceMesh : scaledMeshesOnGPU) {
        SpinImage::gpu::freeMesh(deviceMesh);
    }

    std::cout << std::endl << "Complete." << std::endl;
}



Histogram computeSearchResultHistogram(size_t vertexCount, const SpinImage::array<unsigned int> &searchResults) {

    Histogram histogram;

    float average = 0;
    unsigned int lowerRanks[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t image = 0; image < vertexCount; image++) {
        unsigned int rank = searchResults.content[image];
        histogram.count(rank);
        average += (float(rank) - average) / float(image + 1);

        if(rank < 10) {
            lowerRanks[rank]++;
        }
    }

    std::cout << "\t\tTop 10 counts: ";
    int top10Count = 0;
    for(int i = 0; i < 10; i++) {
        std::cout << lowerRanks[i] << ((i < 9) ? ", " : "");
        top10Count += lowerRanks[i];
    }
    std::cout << " -> average: " << average << ", (" << (double(lowerRanks[0]) / double(vertexCount))*100.0 << "% at rank 0, " << (double(top10Count) / double(vertexCount))*100.0 << "% in top 10)" << std::endl;


    return histogram;
}