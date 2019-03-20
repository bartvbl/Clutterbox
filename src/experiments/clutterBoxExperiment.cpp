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

#include "clutterBox/clutterBoxKernels.cuh"

#include "experimentUtilities/listDir.h"


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

Histogram computeSearchResultHistogram(size_t vertexCount, const array<size_t> &searchResults);

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

void dumpResultsFile(std::string outputFile, size_t seed, std::vector<Histogram> QSIHistograms, std::vector<Histogram> SIHistograms, const std::string &sourceFileDirectory, unsigned int sampleSetSize, float boxSize, float spinImageWidth, size_t assertionRandomToken) {
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
    outFile << "v2" << std::endl;
    outFile << "seed: " << seed << std::endl;
    outFile << "sampleSetSize: " << sampleSetSize << std::endl;
    outFile << "boxSize: " << boxSize << std::endl;
    outFile << "spinImageWidth: " << spinImageWidth << std::endl;
    outFile << "searchResultCount: " << SEARCH_RESULT_COUNT << std::endl;
    for(const std::string &file : chosenFiles) {
        outFile << "file: " << file << std::endl;
    }
    for(const HostMesh &mesh : sampleMeshes) {
        outFile << "vertexCount: " << mesh.vertexCount << std::endl;
    }
    for(const glm::vec3 &rotation : rotations) {
        outFile << "rotation: " << rotation.x << ", " << rotation.y << ", " << rotation.z << std::endl;
    }
    for(const glm::vec3 &translation : translations) {
        outFile << "translation: " << translation.x << ", " << translation.y << ", " << translation.z << std::endl;
    }
    for(unsigned int i = 0; i < sampleSetSize; i++) {
        outFile << "QSIHistogram " << i << ": " << std::endl;
        outFile << QSIHistograms.at(i).toJSON() << std::endl;
    }
    for(unsigned int i = 0; i < sampleSetSize; i++) {
        outFile << "SIHistogram " << i << ": " << std::endl;
        outFile << SIHistograms.at(i).toJSON() << std::endl;
    }


    outFile.close();

    for (unsigned int i = 0; i < sampleSetSize; i++) {
        SpinImage::cpu::freeHostMesh(sampleMeshes.at(i));
    }
}



void runClutterBoxExperiment(cudaDeviceProp device_information, std::string objectDirectory, unsigned int sampleSetSize, float boxSize, unsigned int experimentRepetitions, float spinImageWidth) {
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

    for(unsigned int experiment = 0; experiment < experimentRepetitions; experiment++) {

        std::vector<Histogram> QSIHistograms;
        std::vector<Histogram> spinImageHistograms;

        std::cout << "Selecting file sample set.." << std::endl;

        std::random_device rd;
        size_t randomSeed = 52;//rd();
        std::default_random_engine generator{randomSeed};
        // 1 Search SHREC directory for files
        // 2 Make a sample set of n sample objects
        std::vector<std::string> filePaths = generateRandomFileList(objectDirectory, sampleSetSize, generator);

        // 3 Load the models in the sample set
        std::cout << "Loading sample models.." << std::endl;
        std::vector<HostMesh> sampleMeshes(sampleSetSize);
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            sampleMeshes.at(i) = SpinImage::utilities::loadOBJ(filePaths.at(i), true);
        }

        // 4 Scale all models to fit in a 1x1x1 sphere
        std::cout << "Scaling meshes.." << std::endl;
        std::vector<HostMesh> scaledMeshes(sampleSetSize);
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            scaledMeshes.at(i) = fitMeshInsideSphereOfRadius(sampleMeshes.at(i), 1);
            SpinImage::cpu::freeHostMesh(sampleMeshes.at(i));
        }

        // 5 Copy meshes to GPU
        std::vector<DeviceMesh> scaledMeshesOnGPU(sampleSetSize);
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            scaledMeshesOnGPU.at(i) = SpinImage::copy::hostMeshToDevice(scaledMeshes.at(i));
        }



        std::cout << "Running experiment iteration " << (experiment + 1) << "/" << experimentRepetitions << std::endl;

        // Shuffle the list. First mesh is now our "reference".
        std::shuffle(std::begin(scaledMeshesOnGPU), std::end(scaledMeshesOnGPU), generator);

        size_t spinImageSampleCount = std::max((size_t)1000000, (size_t) (30 * scaledMeshesOnGPU.at(0).vertexCount));
        std::cout << "\tUsing sample count: " << spinImageSampleCount << std::endl;

        // Compute spin image for reference model
        std::cout << "\tGenerating reference QSI images.. (" << scaledMeshesOnGPU.at(0).vertexCount << " images)" << std::endl;
        array<quasiSpinImagePixelType> device_referenceQSIImages = SpinImage::gpu::generateQuasiSpinImages(scaledMeshesOnGPU.at(0),
                                                                                         device_information,
                                                                                         spinImageWidth);
        std::cout << "\tGenerating reference spin images.." << std::endl;
        array<spinImagePixelType> device_referenceSpinImages = SpinImage::gpu::generateSpinImages(scaledMeshesOnGPU.at(0),
                                                                                         device_information,
                                                                                         spinImageWidth,
                                                                                         spinImageSampleCount);
        // Combine meshes into one larger scene
        DeviceMesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

        // Randomly transform objects
        randomlyTransformMeshes(boxScene, boxSize, scaledMeshesOnGPU, generator);

        size_t vertexCount = 0;
        size_t referenceMeshVertexCount = scaledMeshesOnGPU.at(0).vertexCount;
        array<spinImagePixelType> hostReferenceImages = SpinImage::copy::spinImageDescriptorsToHost(device_referenceSpinImages, referenceMeshVertexCount);

        // Generate images for increasingly more complex scenes
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            std::cout << "\tProcessing mesh sample " << (i + 1) << "/" << sampleSetSize << std::endl;
            // Making the generation algorithm believe the scene is smaller than it really is
            vertexCount += scaledMeshesOnGPU.at(i).vertexCount;
            boxScene.vertexCount = vertexCount;

            // Generating images
            std::cout << "\t\tGenerating QSI images.. (" << vertexCount << " images)" << std::endl;
            array<quasiSpinImagePixelType> device_sampleQSIImages = SpinImage::gpu::generateQuasiSpinImages(boxScene,
                                                                                          device_information,
                                                                                          spinImageWidth);


            // Comparing them to the reference ones
            array<size_t> QSIsearchResults = SpinImage::gpu::computeSearchResultRanks(
                    device_referenceQSIImages,
                    referenceMeshVertexCount,
                    device_sampleQSIImages,
                    vertexCount);
            Histogram QSIHistogram = computeSearchResultHistogram(referenceMeshVertexCount, QSIsearchResults);
            std::cout << QSIHistogram.toJSON() << std::endl;
            cudaFree(device_sampleQSIImages.content);
            delete[] QSIsearchResults.content;



            std::cout << "\t\tGenerating spin images.. (" << vertexCount << " images)" << std::endl;
            array<spinImagePixelType> device_sampleSpinImages = SpinImage::gpu::generateSpinImages(boxScene,
                                                                                          device_information,
                                                                                          spinImageWidth,
                                                                                          spinImageSampleCount);
            array<spinImagePixelType> hostDescriptors = SpinImage::copy::spinImageDescriptorsToHost(device_sampleSpinImages, std::min(int(referenceMeshVertexCount), 1000));
            std::stringstream ss;
            SpinImage::dump::searchResults(SpinImage::cpu::findDescriptorsInHaystack(hostReferenceImages, referenceMeshVertexCount, hostDescriptors, vertexCount), "out_cpu.txt");
            hostDescriptors.length = std::min(int(referenceMeshVertexCount), 1000);
            ss << "spinImage_" << i << ".png";
            SpinImage::dump::descriptors(hostDescriptors, ss.str(), true, 50);
            delete[] hostDescriptors.content;
            array<size_t> SpinImageSearchResults = SpinImage::gpu::computeSearchResultRanks(
                    device_referenceSpinImages,
                    referenceMeshVertexCount,
                    device_sampleSpinImages,
                    vertexCount);
            Histogram SIHistogram = computeSearchResultHistogram(referenceMeshVertexCount, SpinImageSearchResults);
            std::cout << SIHistogram.toJSON() << std::endl;
            cudaFree(device_sampleSpinImages.content);
            delete[] SpinImageSearchResults.content;

            QSIHistograms.push_back(QSIHistogram);
            spinImageHistograms.push_back(SIHistogram);

        }

        freeDeviceMesh(boxScene);
        cudaFree(device_referenceQSIImages.content);
        cudaFree(device_referenceSpinImages.content);

        dumpResultsFile("../output/" + getCurrentDateTimeString() + ".txt", randomSeed, QSIHistograms, spinImageHistograms, objectDirectory, sampleSetSize, boxSize, spinImageWidth, generator());
    }
}



Histogram computeSearchResultHistogram(size_t vertexCount, const array<size_t> &searchResults) {

    Histogram histogram;

    float average = 0;

    for (size_t image = 0; image < vertexCount; image++) {
        size_t rank = searchResults.content[image];
        histogram.count(rank);
        average += (float(rank) - average) / float(image + 1);
    }

    std::cout << "\t\tITERATION AVERAGE: " << average << std::endl;

    return histogram;
}