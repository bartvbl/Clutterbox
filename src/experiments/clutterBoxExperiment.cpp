#include "clutterBoxExperiment.hpp"

#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <algorithm>

#include <utilities/stringUtils.h>
#include <utilities/modelScaler.h>
#include <utilities/Histogram.h>

#include <spinImage/cpu/types/HostMesh.h>
#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/cpu/MSIGenerator.h>
#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/gpu/quasiSpinImageGenerator.cuh>
#include <spinImage/gpu/quasiSpinImageSearcher.cuh>
#include <spinImage/gpu/spinImageGenerator.cuh>
#include <spinImage/gpu/spinImageSearcher.cuh>
#include <spinImage/utilities/copy/deviceDescriptorsToHost.h>
#include <spinImage/utilities/dumpers/spinImageDumper.h>
#include <spinImage/utilities/dumpers/searchResultDumper.h>

#include <experiments/clutterBox/clutterBoxUtilities.h>
#include <fstream>
#include <glm/vec3.hpp>
#include <map>
#include <sstream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <spinImage/utilities/spinOriginBufferGenerator.h>

#include "clutterBox/clutterBoxKernels.cuh"

#include "experimentUtilities/listDir.h"
#include "../../../libShapeSearch/lib/nvidia-samples-common/nvidia/helper_cuda.h"


// TODO list:
// - The measure's independent variable should not be number of objects, but rather the number of triangles in the scene
// - How do I manage samples in the scene for spin images? Certain number of samples per triangle?
// - What is the effect of different spin image sizes?
// - In order to limit VRAM usage, as well as get a better signal to noise ratio (due to aliasing) on the images, we should only use models with less than a certain number of vertices.
// -

std::string getCurrentDateTimeString() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%d-%m-%Y %H-%M-%S", timeinfo);
    return std::string(buffer);

}

Histogram computeSearchResultHistogram(size_t vertexCount, const array<unsigned int> &searchResults);

std::vector<std::string> generateRandomFileList(const std::string &objectDirectory, unsigned int sampleSetSize,
                                                std::default_random_engine &generator) {

    std::vector<std::string> filePaths(sampleSetSize);

    std::cout << "Listing object directory..";
    std::vector<std::string> fileList = listDir(objectDirectory);
    std::cout << " (found " << fileList.size() << " files)" << std::endl;

    std::shuffle(std::begin(fileList), std::end(fileList), generator);

    for (unsigned int i = 0; i < sampleSetSize; i++) {
        filePaths[i] = objectDirectory + (endsWith(objectDirectory, "/") ? "" : "/") + fileList.at(i);
        std::cout << "Sample " << i << ": " << filePaths.at(i) << std::endl;
    }

    return filePaths;
}

void dumpResultsFile(
        std::string outputFile,
        size_t seed,
        std::vector<Histogram> QSIHistograms,
        std::vector<Histogram> SIHistograms,
        const std::string &sourceFileDirectory,
        unsigned int sampleSetSize,
        float boxSize,
        float spinImageWidth,
        size_t assertionRandomToken,
        std::vector<SpinImage::debug::QSIRunInfo> QSIRuns,
        std::vector<SpinImage::debug::SIRunInfo> SIRuns,
        std::vector<SpinImage::debug::QSISearchRunInfo> QSISearchRuns,
        std::vector<SpinImage::debug::SISearchRunInfo> SISearchRuns,
        float spinImageSupportAngleDegrees) {
    std::cout << "Dumping results file.." << std::endl;

    std::default_random_engine generator{seed};

    std::vector<std::string> chosenFiles = generateRandomFileList(sourceFileDirectory, sampleSetSize, generator);

    std::shuffle(std::begin(chosenFiles), std::end(chosenFiles), generator);

    std::uniform_real_distribution<float> distribution(0, 1);

    std::vector<glm::vec3> rotations(sampleSetSize);
    std::vector<glm::vec3> translations(sampleSetSize);

    for(unsigned int i = 0; i < sampleSetSize; i++) {
        float yaw = float(distribution(generator) * 2.0 * M_PI);
        float pitch = float((distribution(generator) - 0.5) * M_PI);
        float roll = float(distribution(generator) * 2.0 * M_PI);

        float distanceX = boxSize * distribution(generator);
        float distanceY = boxSize * distribution(generator);
        float distanceZ = boxSize * distribution(generator);

        rotations.at(i) = glm::vec3(yaw, pitch, roll);
        translations.at(i) = glm::vec3(distanceX, distanceY, distanceZ);

        std::cout << "Rotation: (" << yaw << ", " << pitch << ", "<< roll << "), Translation: (" << distanceX << ", "<< distanceY << ", "<< distanceZ << ")" << std::endl;
    }

    std::vector<HostMesh> sampleMeshes(sampleSetSize);
    for (unsigned int i = 0; i < sampleSetSize; i++) {
        sampleMeshes.at(i) = SpinImage::utilities::loadOBJ(chosenFiles.at(i), true);
    }

    size_t finalCheckToken = generator();
    if(finalCheckToken != assertionRandomToken) {
        std::cerr << "ERROR: the verification token generated by the metadata dump function was different than the one used to generate the program output. This means that due to changes the metadata computed by the dump function is likely wrong and should be corrected." << std::endl;
        std::cerr << "Expected: " << finalCheckToken << ", got: " << assertionRandomToken << std::endl;
    }

    std::ofstream outFile(outputFile);
    outFile << "{" << std::endl;
    outFile << "\t\"version\": \"v7\"," << std::endl;
    outFile << "\t\"seed\": " << seed << "," << std::endl;
    outFile << "\t\"sampleSetSize\": " << sampleSetSize << "," << std::endl;
    outFile << "\t\"boxSize\": " << boxSize << "," << std::endl;
    outFile << "\t\"spinImageWidth\": " << spinImageWidth << "," << std::endl;
    outFile << "\t\"spinImageWidthPixels\": " << spinImageWidthPixels << "," << std::endl;
    outFile << "\t\"spinImageSupportAngle\": " << spinImageSupportAngleDegrees << "," << std::endl;
    outFile << "\t\"searchResultCount\": " << SEARCH_RESULT_COUNT << "," << std::endl;
    outFile << std::endl;
    outFile << "\t\"inputFiles\": [" << std::endl;
    for(unsigned int i = 0; i < chosenFiles.size(); i++) {
        outFile << "\t\t\"" << chosenFiles.at(i) << "\"" << ((i == chosenFiles.size() -1) ? "" : ", ") << std::endl;
    }
    outFile << "\t]," << std::endl << std::endl;
    outFile << "\t\"vertexCounts\": [";
    for(unsigned int i = 0; i < sampleMeshes.size(); i++) {
        outFile << sampleMeshes.at(i).vertexCount << ((i == sampleMeshes.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << std::endl;
    outFile << "\t\"rotations\": [" << std::endl;
    for(unsigned int i = 0; i < rotations.size(); i++) {
        outFile << "\t\t[" << rotations.at(i).x << ", " << rotations.at(i).y << ", " << rotations.at(i).z << "]" << ((i == rotations.size() -1) ? "" : ",") << std::endl;
    }
    outFile << "\t]," << std::endl << std::endl;
    outFile << "\t\"translations\": [" << std::endl;
    for(unsigned int i = 0; i < translations.size(); i++) {
        outFile << "\t\t[" << translations.at(i).x << ", " << translations.at(i).y << ", " << translations.at(i).z << "]" << ((i == translations.size() -1) ? "" : ",") << std::endl;
    }
    outFile << "\t]," << std::endl << std::endl;
    outFile << "\t\"runtimes\": {" << std::endl;

    outFile << "\t\t\"QSIReferenceGeneration\": {" << std::endl;
    outFile << "\t\t\t\"total\": " << QSIRuns.at(0).totalExecutionTimeSeconds << ", " << std::endl;
    outFile << "\t\t\t\"meshScale\": " << QSIRuns.at(0).meshScaleTimeSeconds << ", " << std::endl;
    outFile << "\t\t\t\"redistribution\": " << QSIRuns.at(0).redistributionTimeSeconds << ", " << std::endl;
    outFile << "\t\t\t\"generation\": " << QSIRuns.at(0).generationTimeSeconds << std::endl;
    outFile << "\t\t}," << std::endl << std::endl;

    outFile << "\t\t\"SIReferenceGeneration\": {" << std::endl;
    outFile << "\t\t\t\"total\": " << SIRuns.at(0).totalExecutionTimeSeconds << ", " << std::endl;
    outFile << "\t\t\t\"initialisation\": " << SIRuns.at(0).initialisationTimeSeconds<< ", " << std::endl;
    outFile << "\t\t\t\"sampling\": " << SIRuns.at(0).meshSamplingTimeSeconds << ", " << std::endl;
    outFile << "\t\t\t\"generation\": " << SIRuns.at(0).generationTimeSeconds << std::endl;
    outFile << "\t\t}," << std::endl << std::endl;

    outFile << "\t\t\"QSISampleGeneration\": {" << std::endl;
    outFile << "\t\t\t\"total\": [";
    for(unsigned int i = 1; i < QSIRuns.size(); i++) {
        outFile << QSIRuns.at(i).totalExecutionTimeSeconds << ((i == QSIRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"meshScale\": [";
    for(unsigned int i = 1; i < QSIRuns.size(); i++) {
        outFile << QSIRuns.at(i).meshScaleTimeSeconds << ((i == QSIRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"redistribution\": [";
    for(unsigned int i = 1; i < QSIRuns.size(); i++) {
        outFile << QSIRuns.at(i).redistributionTimeSeconds << ((i == QSIRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"generation\": [";
    for(unsigned int i = 1; i < QSIRuns.size(); i++) {
        outFile << QSIRuns.at(i).generationTimeSeconds << ((i == QSIRuns.size() -1) ? "" : ", ");
    }
    outFile << "]" << std::endl;
    outFile << "\t\t}," << std::endl << std::endl;

    outFile << "\t\t\"SISampleGeneration\": {" << std::endl;
    outFile << "\t\t\t\"total\": [";
    for(unsigned int i = 1; i < SIRuns.size(); i++) {
        outFile << SIRuns.at(i).totalExecutionTimeSeconds << ((i == SIRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"initialisation\": [";
    for(unsigned int i = 1; i < SIRuns.size(); i++) {
        outFile << SIRuns.at(i).initialisationTimeSeconds << ((i == SIRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"sampling\": [";
    for(unsigned int i = 1; i < SIRuns.size(); i++) {
        outFile << SIRuns.at(i).meshSamplingTimeSeconds << ((i == SIRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"generation\": [";
    for(unsigned int i = 1; i < SIRuns.size(); i++) {
        outFile << SIRuns.at(i).generationTimeSeconds << ((i == SIRuns.size() -1) ? "" : ", ");
    }
    outFile << "]" << std::endl;
    outFile << "\t\t}," << std::endl << std::endl;

    outFile << "\t\t\"QSISearch\": {" << std::endl;
    outFile << "\t\t\t\"total\": [";
    for(unsigned int i = 0; i < QSISearchRuns.size(); i++) {
        outFile << QSISearchRuns.at(i).totalExecutionTimeSeconds << ((i == QSISearchRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"search\": [";
    for(unsigned int i = 0; i < QSISearchRuns.size(); i++) {
        outFile << QSISearchRuns.at(i).searchExecutionTimeSeconds << ((i == QSISearchRuns.size() -1) ? "" : ", ");
    }
    outFile << "]" << std::endl;
    outFile << "\t\t}," << std::endl << std::endl;

    outFile << "\t\t\"SISearch\": {" << std::endl;
    outFile << "\t\t\t\"total\": [";
    for(unsigned int i = 0; i < SISearchRuns.size(); i++) {
        outFile << SISearchRuns.at(i).totalExecutionTimeSeconds << ((i == SISearchRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"averaging\": [";
    for(unsigned int i = 0; i < SISearchRuns.size(); i++) {
        outFile << SISearchRuns.at(i).averagingExecutionTimeSeconds << ((i == SISearchRuns.size() -1) ? "" : ", ");
    }
    outFile << "]," << std::endl << "\t\t\t\"search\": [";
    for(unsigned int i = 0; i < SISearchRuns.size(); i++) {
        outFile << SISearchRuns.at(i).searchExecutionTimeSeconds << ((i == SISearchRuns.size() -1) ? "" : ", ");
    }
    outFile << "]" << std::endl;
    outFile << "\t\t}" << std::endl;

    outFile << "\t}," << std::endl;

    outFile << std::endl << "\t\"QSIhistograms\": [" << std::endl;
    for(unsigned int i = 0; i < sampleSetSize; i++) {
        outFile << QSIHistograms.at(i).toJSON(2) << ((i == sampleSetSize -1) ? "" : ", ") << std::endl;
    }
    outFile << "\t]," << std::endl << "\t\"SIhistograms\": [" << std::endl;
    for(unsigned int i = 0; i < sampleSetSize; i++) {
        outFile << SIHistograms.at(i).toJSON(2) << ((i == sampleSetSize -1) ? "" : ", ") << std::endl;
    }
    outFile << "\t]" << std::endl;
    outFile << "}" << std::endl;

    outFile.close();

    for (unsigned int i = 0; i < sampleSetSize; i++) {
        SpinImage::cpu::freeHostMesh(sampleMeshes.at(i));
    }
}


const inline size_t computeSpinImageSampleCount(size_t &vertexCount) {
    return std::max((size_t)1000000, (size_t) (30 * vertexCount)); 
}

void dumpSpinImages(std::string filename, array<spinImagePixelType> device_descriptors) {
    size_t arrayLength = std::min(device_descriptors.length, (size_t)2500);
    array<float> hostDescriptors = SpinImage::copy::spinImageDescriptorsToHost(device_descriptors, arrayLength);
    hostDescriptors.length = arrayLength;
    SpinImage::dump::descriptors(hostDescriptors, filename, true, 50);
    delete[] hostDescriptors.content;
}

void dumpQuasiSpinImages(std::string filename, array<quasiSpinImagePixelType> device_descriptors) {
    size_t arrayLength = std::min(device_descriptors.length, (size_t)2500);
    array<quasiSpinImagePixelType > hostDescriptors = SpinImage::copy::QSIDescriptorsToHost(device_descriptors, std::min(device_descriptors.length, (size_t)2500));
    hostDescriptors.length = arrayLength;
    SpinImage::dump::descriptors(hostDescriptors, filename, true, 50);
    delete[] hostDescriptors.content;
}

void runClutterBoxExperiment(std::string objectDirectory, unsigned int sampleSetSize, float boxSize, float spinImageWidth, float spinImageSupportAngleDegrees, size_t overrideSeed) {
	// --- Overview ---
	//
	// 1 Search SHREC directory for files
	// 2 Make a sample set of n sample objects
	// 3 Load the models in the sample set
	// 4 Scale all models to fit in a 1x1x1 sphere
	// 5 Compute (quasi) spin images for all models in the sample set
	// 6 Create a box of SxSxS units
	// 7 for all combinations (non-reused) models:
	//    7.1 Place each mesh in the box, retrying if it collides with another mesh
	//    7.2 For all meshes in the box, compute spin images for all vertices
	//    7.3 Compare the generated images to the "clutter-free" variants
	//    7.4 Dump the distances between images

    std::vector<Histogram> QSIHistograms;
    std::vector<Histogram> spinImageHistograms;
    std::vector<SpinImage::debug::QSIRunInfo> QSIRuns;
    std::vector<SpinImage::debug::SIRunInfo> SIRuns;
    std::vector<SpinImage::debug::SISearchRunInfo> SISearchRuns;
    std::vector<SpinImage::debug::QSISearchRunInfo> QSISearchRuns;

    std::cout << "Selecting file sample set.." << std::endl;

    std::random_device rd;
    size_t randomSeed = overrideSeed != 0 ? overrideSeed : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::default_random_engine generator{randomSeed};
    // 1 Search SHREC directory for files
    // 2 Make a sample set of n sample objects
    std::vector<std::string> filePaths = generateRandomFileList(objectDirectory, sampleSetSize, generator);

    // 3 Load the models in the sample set
    std::cout << "Loading sample models.." << std::endl;
    std::vector<HostMesh> sampleMeshes(sampleSetSize);
    for (unsigned int i = 0; i < sampleSetSize; i++) {
        sampleMeshes.at(i) = SpinImage::utilities::loadOBJ(filePaths.at(i), true);
        std::cout << "\tMesh " << i << ": " << sampleMeshes.at(i).vertexCount << " vertices" << std::endl;
    }

    // 4 Scale all models to fit in a 1x1x1 sphere
    std::cout << "Scaling meshes.." << std::endl;
    std::vector<HostMesh> scaledMeshes(sampleSetSize);
    for (unsigned int i = 0; i < sampleSetSize; i++) {
        scaledMeshes.at(i) = fitMeshInsideSphereOfRadius(sampleMeshes.at(i), 1);
        SpinImage::cpu::freeHostMesh(sampleMeshes.at(i));
    }

    // 5 Copy meshes to GPU
    std::cout << "Copying meshes to device.." << std::endl;
    std::vector<DeviceMesh> scaledMeshesOnGPU(sampleSetSize);
    for (unsigned int i = 0; i < sampleSetSize; i++) {
        scaledMeshesOnGPU.at(i) = SpinImage::copy::hostMeshToDevice(scaledMeshes.at(i));
    }

    // 6 Remove duplicate vertices
    std::cout << "Removing duplicate vertices.." << std::endl;
    array<DeviceOrientedPoint> spinOrigins_reference = SpinImage::utilities::generateUniqueSpinOriginBuffer(scaledMeshesOnGPU.at(0));
    size_t referenceImageCount = spinOrigins_reference.length;

    // Shuffle the list. First mesh is now our "reference".
    std::shuffle(std::begin(scaledMeshesOnGPU), std::end(scaledMeshesOnGPU), generator);

    size_t spinImageSampleCount = computeSpinImageSampleCount(scaledMeshesOnGPU.at(0).vertexCount);
    std::cout << "\tUsing sample count: " << spinImageSampleCount << std::endl;

    // Compute spin image for reference model
    std::cout << "\tGenerating reference QSI images.. (" << referenceImageCount << " images)" << std::endl;
    SpinImage::debug::QSIRunInfo qsiReferenceRunInfo;
    array<quasiSpinImagePixelType> device_referenceQSIImages = SpinImage::gpu::generateQuasiSpinImages(
                                                                                     scaledMeshesOnGPU.at(0),
                                                                                     spinOrigins_reference,
                                                                                     spinImageWidth,
                                                                                     &qsiReferenceRunInfo);
    //dumpQuasiSpinImages("qsi_verification.png", device_referenceQSIImages);
    QSIRuns.push_back(qsiReferenceRunInfo);
    std::cout << "\t\tExecution time: " << qsiReferenceRunInfo.generationTimeSeconds << std::endl;

    std::cout << "\tGenerating reference spin images.." << std::endl;
    SpinImage::debug::SIRunInfo siReferenceRunInfo;
    array<spinImagePixelType> device_referenceSpinImages = SpinImage::gpu::generateSpinImages(
                                                                                     scaledMeshesOnGPU.at(0),
                                                                                     spinOrigins_reference,
                                                                                     spinImageWidth,
                                                                                     spinImageSampleCount,
                                                                                     spinImageSupportAngleDegrees,
                                                                                     &siReferenceRunInfo);

    checkCudaErrors(cudaFree(spinOrigins_reference.content));

    //dumpSpinImages("si_verification.png", device_referenceSpinImages);
    SIRuns.push_back(siReferenceRunInfo);
    std::cout << "\t\tExecution time: " << siReferenceRunInfo.generationTimeSeconds << std::endl;

    // Combine meshes into one larger scene
    DeviceMesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

    // Randomly transform objects
    randomlyTransformMeshes(boxScene, boxSize, scaledMeshesOnGPU, generator);

    size_t vertexCount = 0;
    size_t referenceMeshVertexCount = scaledMeshesOnGPU.at(0).vertexCount;
    size_t referenceMeshImageCount = spinOrigins_reference.length;



    // Generate images for increasingly more complex scenes
    for (unsigned int i = 0; i < sampleSetSize; i++) {
        std::cout << "\tProcessing mesh sample " << (i + 1) << "/" << sampleSetSize << std::endl;
        // Making the generation algorithm believe the scene is smaller than it really is
        // This allows adding objects one by one, without having to copy memory all over the place
        vertexCount += scaledMeshesOnGPU.at(i).vertexCount;
        boxScene.vertexCount = vertexCount;

        array<DeviceOrientedPoint> spinOrigins_sample = SpinImage::utilities::generateUniqueSpinOriginBuffer(boxScene);
        size_t imageCount = spinOrigins_sample.length;

        // Generating quasi spin images
        std::cout << "\t\tGenerating QSI images.. (" << imageCount << " images)" << std::endl;
        SpinImage::debug::QSIRunInfo qsiSampleRunInfo;
        array<quasiSpinImagePixelType> device_sampleQSIImages = SpinImage::gpu::generateQuasiSpinImages(
                boxScene,
                spinOrigins_sample,
                spinImageWidth,
                &qsiSampleRunInfo);
        QSIRuns.push_back(qsiSampleRunInfo);
        std::cout << "\t\t\tTimings: (total " << qsiSampleRunInfo.totalExecutionTimeSeconds
                  << ", scaling " << qsiSampleRunInfo.meshScaleTimeSeconds
                  << ", redistribution " << qsiSampleRunInfo.redistributionTimeSeconds
                  << ", generation " << qsiSampleRunInfo.generationTimeSeconds << ")" << std::endl;

        std::cout << "\t\tSearching in quasi spin images.." << std::endl;
        SpinImage::debug::QSISearchRunInfo qsiSearchRun;
        array<unsigned int> QSIsearchResults = SpinImage::gpu::computeQuasiSpinImageSearchResultRanks(
                device_referenceQSIImages,
                referenceMeshImageCount,
                device_sampleQSIImages,
                imageCount,
                &qsiSearchRun);
        QSISearchRuns.push_back(qsiSearchRun);
        std::cout << "\t\t\tTimings: (total " << qsiSearchRun.totalExecutionTimeSeconds
                  << ", searching " << qsiSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
        Histogram QSIHistogram = computeSearchResultHistogram(referenceMeshImageCount, QSIsearchResults);
        cudaFree(device_sampleQSIImages.content);
        delete[] QSIsearchResults.content;



        // Generating spin images
        spinImageSampleCount = computeSpinImageSampleCount(imageCount);
        std::cout << "\t\tGenerating spin images.. (" << imageCount << " images, " << spinImageSampleCount << " samples)" << std::endl;
        SpinImage::debug::SIRunInfo siSampleRunInfo;
        array<spinImagePixelType> device_sampleSpinImages = SpinImage::gpu::generateSpinImages(
                boxScene,
                spinOrigins_sample,
                spinImageWidth,
                spinImageSampleCount,
                spinImageSupportAngleDegrees,
                &siSampleRunInfo);
        SIRuns.push_back(siSampleRunInfo);
        std::cout << "\t\t\tTimings: (total " << siSampleRunInfo.totalExecutionTimeSeconds
                  << ", initialisation " << siSampleRunInfo.initialisationTimeSeconds
                  << ", sampling " << siSampleRunInfo.meshSamplingTimeSeconds
                  << ", generation " << siSampleRunInfo.generationTimeSeconds << ")" << std::endl;

        std::cout << "\t\tSearching in spin images.." << std::endl;
        SpinImage::debug::SISearchRunInfo siSearchRun;
        array<unsigned int> SpinImageSearchResults = SpinImage::gpu::computeSpinImageSearchResultRanks(
                device_referenceSpinImages,
                referenceMeshImageCount,
                device_sampleSpinImages,
                imageCount,
                &siSearchRun);
        SISearchRuns.push_back(siSearchRun);
        std::cout << "\t\t\tTimings: (total " << siSearchRun.totalExecutionTimeSeconds
                  << ", averaging " << siSearchRun.averagingExecutionTimeSeconds
                  << ", searching " << siSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
        Histogram SIHistogram = computeSearchResultHistogram(referenceMeshImageCount, SpinImageSearchResults);
        cudaFree(device_sampleSpinImages.content);
        delete[] SpinImageSearchResults.content;


        // Cleaning up
        checkCudaErrors(cudaFree(spinOrigins_sample.content));

        // Storing results
        QSIHistograms.push_back(QSIHistogram);
        spinImageHistograms.push_back(SIHistogram);

    }

    SpinImage::gpu::freeDeviceMesh(boxScene);
    cudaFree(device_referenceQSIImages.content);
    cudaFree(device_referenceSpinImages.content);

    dumpResultsFile("../output/" + getCurrentDateTimeString() + ".json", randomSeed, QSIHistograms, spinImageHistograms, objectDirectory, sampleSetSize, boxSize, spinImageWidth, generator(), QSIRuns, SIRuns, QSISearchRuns, SISearchRuns, spinImageSupportAngleDegrees);

}



Histogram computeSearchResultHistogram(size_t vertexCount, const array<unsigned int> &searchResults) {

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

    std::cout << "\t\t\tTop 10 counts: ";
    int top10Count = 0;
    for(int i = 0; i < 10; i++) {
        std::cout << lowerRanks[i] << ((i < 9) ? ", " : "");
        top10Count += lowerRanks[i];
    }
    std::cout << " -> average: " << average << ", (" << (double(lowerRanks[0]) / double(vertexCount))*100.0 << "% at rank 0, " << (double(top10Count) / double(vertexCount))*100.0 << "% in top 10)" << std::endl;


    return histogram;
}