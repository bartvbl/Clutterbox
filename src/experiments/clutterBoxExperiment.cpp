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
#include <shapeSearch/gpu/quasiSpinImageSearcher.cuh>
#include <experiments/clutterBox/clutterBoxUtilities.h>
#include <fstream>

#include "clutterBox/clutterBoxKernels.cuh"

#include "experimentUtilities/listDir.h"


// TODO list:
// - implementation that searches in the clutter box images for reference images
//      - Create a histogram based on at which rank each image appears
// - The measure's independent variable should not be number of objects, but rather the number of triangles in the scene
// - How do I manage samples in the scene for spin images? Certain number of samples per triangle?
// - What is the effect of different spin image sizes?
// - In order to limit VRAM usage, as well as get a better signal to noise ratio (due to aliasing) on the images, we should only use models with less than a certain number of vertices.
// -



bool contains(std::vector<unsigned int> &haystack, unsigned int needle);

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

    // 2 Make a sample set of n sample objects
    std::cout << "Selecting file sample set.." << std::endl;
    std::random_device rd;
    std::default_random_engine generator(40);//{rd()};
	std::uniform_int_distribution<unsigned int> distribution(0, fileList.size());

	std::vector<unsigned int> sampleIndices(sampleSetSize);

	for(unsigned int i = 0; i < sampleSetSize; i++) {
		unsigned int randomIndex;
		do {
			randomIndex = distribution(generator);
			sampleIndices[i] = randomIndex;
		} while(!contains(sampleIndices, randomIndex));
	}
	std::sort(sampleIndices.begin(), sampleIndices.end());

	std::vector<std::string> filePaths(sampleSetSize);
	for(unsigned int i = 0; i < sampleSetSize; i++) {
		filePaths[i] = objectDirectory + (endsWith(objectDirectory, "/") ? "" : "/") + fileList.at(sampleIndices[i]);
		std::cout << "Sample: " << sampleIndices[i] << " -> " << filePaths.at(i) << std::endl;
	}

	// 3 Load the models in the sample set
	std::cout << "Loading sample models.." << std::endl;
	std::vector<HostMesh> sampleMeshes(sampleSetSize);
	for(unsigned int i = 0; i < sampleSetSize; i++) {
		sampleMeshes.at(i) = hostLoadOBJ(filePaths.at(i));
	}

	// 4 Scale all models to fit in a 1x1x1 sphere
	std::cout << "Scaling meshes.." << std::endl;
	std::vector<HostMesh> scaledMeshes(sampleSetSize);
	for(unsigned int i = 0; i < sampleSetSize; i++) {
		scaledMeshes.at(i) = scaleMesh(sampleMeshes.at(i), 1);
	}

    // 5 Copy meshes to GPU
	std::vector<DeviceMesh> scaledMeshesOnGPU(sampleSetSize);
	for(unsigned int i = 0; i < sampleSetSize; i++) {
	    scaledMeshesOnGPU.at(i) = copyMeshToGPU(scaledMeshes.at(i));
	}

    for(unsigned int experiment = 0; experiment < experimentRepetitions; experiment++) {
    	std::cout << "Running experiment iteration " << (experiment+1) << "/" << experimentRepetitions << std::endl;

    	// Shuffle the list. First mesh is now our "reference".
        std::shuffle(std::begin(scaledMeshesOnGPU), std::end(scaledMeshesOnGPU), generator);

		// Compute spin image for reference model
		std::cout << "\tGenerating reference QSI images.." << std::endl;
		array<unsigned int> device_referenceImages = generateQuasiSpinImages(scaledMeshesOnGPU.at(0), device_information, spinImageWidth);

        // Combine meshes into one larger scene
        DeviceMesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

		// Randomly transform objects
		randomlyTransformMeshes(boxScene, boxSize, scaledMeshesOnGPU, generator);

		size_t vertexCount = 0;
		size_t referenceMeshVertexCount = scaledMeshesOnGPU.at(0).vertexCount;

		// Generate images for increasingly more complex scenes
		for(unsigned int i = 0; i < sampleSetSize; i++) {
			std::cout << "\tProcessing mesh sample " << (i+1) << "/" << sampleSetSize << std::endl;
			// Making the generation algorithm believe the scene is smaller than it really is
	    	vertexCount += scaledMeshesOnGPU.at(i).vertexCount;
			boxScene.vertexCount = vertexCount;

			// Generating images
			std::cout << "\t\tGenerating QSI images.. (" << vertexCount << " images)" << std::endl;
			array<unsigned int> device_sampleImages = generateQuasiSpinImages(boxScene, device_information, spinImageWidth);

            std::vector<unsigned int> histogram;
            histogram.resize(33);
            std::fill(histogram.begin(), histogram.end(), 0);

			// Comparing them to the reference ones
			array<ImageSearchResults> searchResults = findDescriptorsInHaystack(device_referenceImages, referenceMeshVertexCount, device_sampleImages, vertexCount);

            std::ofstream indicesFile;
            std::ofstream scoresFile;
            indicesFile.open("indices.txt");
            scoresFile.open("scores.txt");

			float average = 0;
			for(size_t image = 0; image < vertexCount; image++) {
			    for(unsigned int i = 0; i < 32; i++) {
                    indicesFile << searchResults.content[image].resultIndices[i] << (i == 31 ? "\r\n" : ", ");
                    scoresFile << searchResults.content[image].resultScores[i] << (i == 31 ? "\r\n" : ", ");
			    }

			    unsigned int topSearchResult = 0;
			    for(; topSearchResult < 32; topSearchResult++) {
                    if(searchResults.content[image].resultIndices[topSearchResult] == image) {
                        break;
                    }
			    }

			    histogram.at(topSearchResult)++;
			    average += (float(topSearchResult) - average) / float(image + 1);
			}

            indicesFile.close();
			scoresFile.close();

			std::cout << "\t\tITERATION AVERAGE: " << average << std::endl;

			for(unsigned int histogramEntry = 0; histogramEntry < histogram.size(); histogramEntry++) {
			    std::cout << "\t\t\t" << histogramEntry << " -> " << histogram.at(histogramEntry) << std::endl;
			}

			cudaFree(device_sampleImages.content);

		}

	    freeDeviceMesh(boxScene);
	    cudaFree(device_referenceImages.content);
    }



}

bool contains(std::vector<unsigned int> &haystack, unsigned int needle) {
	for(unsigned int i = 0; i < haystack.size(); i++) {
		if(haystack[i] == needle) {
			return true;
		}
	}
	return false;
}