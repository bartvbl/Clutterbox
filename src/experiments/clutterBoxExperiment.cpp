#include "clutterBoxExperiment.hpp"

#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <algorithm>

#include <utilities/stringUtils.h>
#include <utilities/modelScaler.h>

#include <shapeSearch/cpu/types/HostMesh.h>
#include <shapeSearch/utilities/OBJLoader.h>
#include <shapeSearch/cpu/MSIGenerator.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>
#include <shapeSearch/gpu/CopyMeshHostToDevice.h>
#include <shapeSearch/gpu/quasiSpinImageGenerator.cuh>
#include <shapeSearch/gpu/spinImageGenerator.cuh>
#include <shapeSearch/gpu/quasiSpinImageSearcher.cuh>
#include <experiments/clutterBox/clutterBoxUtilities.h>
#include <fstream>
#include <shapeSearch/gpu/copyDescriptorsToHost.h>
#include <shapeSearch/utilities/spinImageDumper.h>

#include "clutterBox/clutterBoxKernels.cuh"

#include "experimentUtilities/listDir.h"


// TODO list:
// - The measure's independent variable should not be number of objects, but rather the number of triangles in the scene
// - How do I manage samples in the scene for spin images? Certain number of samples per triangle?
// - What is the effect of different spin image sizes?
// - In order to limit VRAM usage, as well as get a better signal to noise ratio (due to aliasing) on the images, we should only use models with less than a certain number of vertices.
// -



bool contains(std::vector<unsigned int> &haystack, unsigned int needle);

std::vector<unsigned int> computeSearchResultHistogram(size_t vertexCount, const array<ImageSearchResults> &searchResults);

void runClutterBoxExperiment(cudaDeviceProp device_information, std::string objectDirectory, unsigned int sampleSetSize, float boxSize, unsigned int experimentRepetitions, int spinImageWidth) {
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


    // 1 Search SHREC directory for files
    std::cout << "Listing object directory..";
    std::vector<std::string> fileList = listDir(objectDirectory);
    std::cout << " (found " << fileList.size() << " files)" << std::endl;

    for(unsigned int experiment = 0; experiment < experimentRepetitions; experiment++) {

        // 2 Make a sample set of n sample objects
        std::cout << "Selecting file sample set.." << std::endl;
        std::random_device rd;
        std::default_random_engine generator(41);//{rd()};
        std::uniform_int_distribution<unsigned int> distribution(0, fileList.size());

        std::vector<unsigned int> sampleIndices(sampleSetSize);

        for (unsigned int i = 0; i < sampleSetSize; i++) {
            unsigned int randomIndex;
            do {
                randomIndex = distribution(generator);
                sampleIndices[i] = randomIndex;
            } while (!contains(sampleIndices, randomIndex));
        }
        std::sort(sampleIndices.begin(), sampleIndices.end());

        std::vector<std::string> filePaths(sampleSetSize);
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            filePaths[i] =
                    objectDirectory + (endsWith(objectDirectory, "/") ? "" : "/") + fileList.at(sampleIndices[i]);
            std::cout << "Sample: " << sampleIndices[i] << " -> " << filePaths.at(i) << std::endl;
        }

        // 3 Load the models in the sample set
        std::cout << "Loading sample models.." << std::endl;
        std::vector<HostMesh> sampleMeshes(sampleSetSize);
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            sampleMeshes.at(i) = hostLoadOBJ(filePaths.at(i), true);
        }

        // 4 Scale all models to fit in a 1x1x1 sphere
        std::cout << "Scaling meshes.." << std::endl;
        std::vector<HostMesh> scaledMeshes(sampleSetSize);
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            scaledMeshes.at(i) = fitMeshInsideSphereOfRadius(sampleMeshes.at(i), 1);
            freeHostMesh(sampleMeshes.at(i));
        }

        // 5 Copy meshes to GPU
        std::vector<DeviceMesh> scaledMeshesOnGPU(sampleSetSize);
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            scaledMeshesOnGPU.at(i) = copyMeshToGPU(scaledMeshes.at(i));
        }


        std::cout << "Running experiment iteration " << (experiment + 1) << "/" << experimentRepetitions << std::endl;

        // Shuffle the list. First mesh is now our "reference".
        std::shuffle(std::begin(scaledMeshesOnGPU), std::end(scaledMeshesOnGPU), generator);

        // Compute spin image for reference model
        std::cout << "\tGenerating reference QSI images.." << std::endl;
        array<newSpinImagePixelType> device_referenceQSIImages = generateQuasiSpinImages(scaledMeshesOnGPU.at(0),
                                                                                         device_information,
                                                                                         spinImageWidth);
        std::cout << "\tGenerating reference spin images.." << std::endl;
        array<classicSpinImagePixelType> device_referenceSpinImages = generateSpinImages(scaledMeshesOnGPU.at(0),
                                                                                         device_information,
                                                                                         spinImageWidth,
                                                                                         1000000);
        //dumpImages(copySpinImageDescriptorsToHost(device_referenceSpinImages, scaledMeshesOnGPU.at(0).vertexCount), "reference_spin.png", true, 70);

        // Combine meshes into one larger scene
        DeviceMesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

        // Randomly transform objects
        randomlyTransformMeshes(boxScene, boxSize, scaledMeshesOnGPU, generator);

        size_t vertexCount = 0;
        size_t referenceMeshVertexCount = scaledMeshesOnGPU.at(0).vertexCount;

        // Generate images for increasingly more complex scenes
        for (unsigned int i = 0; i < sampleSetSize; i++) {
            std::cout << "\tProcessing mesh sample " << (i + 1) << "/" << sampleSetSize << std::endl;
            // Making the generation algorithm believe the scene is smaller than it really is
            vertexCount += scaledMeshesOnGPU.at(i).vertexCount;
            boxScene.vertexCount = vertexCount;

            // Generating images
            std::cout << "\t\tGenerating QSI images.. (" << vertexCount << " images)" << std::endl;
            array<newSpinImagePixelType> device_sampleQSIImages = generateQuasiSpinImages(boxScene,
                                                                                          device_information,
                                                                                          spinImageWidth);
            std::cout << "\t\tGenerating spin images.. (" << vertexCount << " images)" << std::endl;
            array<classicSpinImagePixelType> device_sampleSpinImages = generateSpinImages(boxScene,
                                                                                          device_information,
                                                                                          spinImageWidth,
                                                                                          1000000);
            //dumpImages(copySpinImageDescriptorsToHost(device_sampleSpinImages, boxScene.vertexCount), "sample_spin.png", true, 70);


            // Comparing them to the reference ones
            array<ImageSearchResults> QSIsearchResults = findDescriptorsInHaystack(
                    device_referenceQSIImages,
                    referenceMeshVertexCount,
                    device_sampleQSIImages,
                    vertexCount);
            array<ImageSearchResults> SpinImageSearchResults = findDescriptorsInHaystack(
                    device_referenceSpinImages,
                    referenceMeshVertexCount,
                    device_sampleSpinImages,
                    vertexCount);

            std::vector<unsigned int> QSIHistogram = computeSearchResultHistogram(referenceMeshVertexCount, QSIsearchResults);
            std::vector<unsigned int> SIHistogram = computeSearchResultHistogram(referenceMeshVertexCount, SpinImageSearchResults);


            for (unsigned int histogramEntry = 0; histogramEntry < QSIHistogram.size(); histogramEntry++) {
                std::cout << "\t\t\t" << histogramEntry << " -> " << QSIHistogram.at(histogramEntry) << "\t\t" << SIHistogram.at(histogramEntry) << std::endl;
            }

            cudaFree(device_sampleQSIImages.content);
            cudaFree(device_sampleSpinImages.content);

            delete[] QSIsearchResults.content;
            delete[] SpinImageSearchResults.content;

        }

        freeDeviceMesh(boxScene);
        cudaFree(device_referenceQSIImages.content);
        cudaFree(device_referenceSpinImages.content);
    }
}

std::vector<unsigned int> computeSearchResultHistogram(size_t vertexCount, const array<ImageSearchResults> &searchResults) {
    std::vector<unsigned int> histogram;
    histogram.resize(SEARCH_RESULT_COUNT + 1);
    std::fill(histogram.begin(), histogram.end(), 0);
    float average = 0;

    for (size_t image = 0; image < vertexCount; image++) {

        float lastEquivalentScore = searchResults.content[image].resultScores[0];
        size_t lastEquivalentIndex = 0;

        unsigned int topSearchResult = 0;
        for (; topSearchResult < SEARCH_RESULT_COUNT; topSearchResult++) {
            float searchResultScore = searchResults.content[image].resultScores[topSearchResult];
            size_t searchResultIndex = searchResults.content[image].resultIndices[topSearchResult];

            if (lastEquivalentScore != searchResultScore) {
                lastEquivalentScore = searchResultScore;
                lastEquivalentIndex = topSearchResult;
            }

            if (searchResultIndex == image) {
                break;
            }
        }

        histogram.at(lastEquivalentIndex)++;
        average += (float(lastEquivalentIndex) - average) / float(image + 1);
    }

    std::cout << "\t\tITERATION AVERAGE: " << average << std::endl;

    return histogram;
}

bool contains(std::vector<unsigned int> &haystack, unsigned int needle) {
	for(unsigned int i = 0; i < haystack.size(); i++) {
		if(haystack[i] == needle) {
			return true;
		}
	}
	return false;
}