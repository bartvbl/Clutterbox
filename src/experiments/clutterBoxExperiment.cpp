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
#include <shapeSearch/gpu/quasiSpinImageComparator.cuh>
#include <experiments/clutterBox/clutterBoxUtilities.h>

#include "clutterBox/clutterBoxKernels.cuh"

#include "experimentUtilities/listDir.h"

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
	std::default_random_engine generator( (unsigned int)time(0) );
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

	std::uniform_real_distribution<float> boxDistribution(0, 1);
    std::random_device rd;
    std::default_random_engine randomGenerator = std::default_random_engine {rd()};

    for(unsigned int experiment = 0; experiment < experimentRepetitions; experiment++) {
    	std::cout << "Running experiment iteration " << experiment << std::endl;

    	// Shuffle the list. First mesh is now our "reference".
        std::shuffle(std::begin(scaledMeshesOnGPU), std::end(scaledMeshesOnGPU), randomGenerator);

		// Compute spin image for reference model
		array<unsigned int> referenceImages = generateQuasiSpinImages(scaledMeshesOnGPU.at(0), device_information, spinImageWidth);

        // Combine meshes into one larger scene
        DeviceMesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

		// Randomly transform objects
		randomlyTransformMeshes(boxScene, scaledMeshesOnGPU, randomGenerator);

	    // Generate images for increasingly more complex scenes
		unsigned int vertexCount = 0;
	    for(unsigned int i = 0; i < sampleSetSize; i++) {
			// Making the generation algorithm believe the scene is smaller than it really is
	    	vertexCount += scaledMeshesOnGPU.at(i).vertexCount;
			boxScene.vertexCount = vertexCount;

			// Generating images
			array<unsigned int> sampleImages = generateQuasiSpinImages(boxScene, device_information, spinImageWidth);

			// Comparing them to the reference ones
			array<float> distances = compareDescriptorsElementWise(referenceImages, sampleImages, vertexCount);

		}

	    freeDeviceMesh(boxScene);
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