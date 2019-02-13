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

#include "clutterBox/clutterBoxKernels.cuh"

#include "experimentUtilities/listDir.h"

bool contains(std::vector<unsigned int> &haystack, unsigned int needle);

void runClutterBoxExperiment(cudaDeviceProp device_information, std::string objectDirectory, int sampleSetSize, float boxSize, int spinImageWidth) {
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

	for(int i = 0; i < sampleSetSize; i++) {
		unsigned int randomIndex;
		do {
			randomIndex = distribution(generator);
			sampleIndices[i] = randomIndex;
		} while(!contains(sampleIndices, randomIndex));
	}
	std::sort(sampleIndices.begin(), sampleIndices.end());

	std::vector<std::string> filePaths(sampleSetSize);
	for(int i = 0; i < sampleSetSize; i++) {
		filePaths[i] = objectDirectory + (endsWith(objectDirectory, "/") ? "" : "/") + fileList.at(sampleIndices[i]);
		std::cout << "Sample: " << sampleIndices[i] << " -> " << filePaths.at(i) << std::endl;
	}

	// 3 Load the models in the sample set
	std::cout << "Loading sample models.." << std::endl;
	std::vector<HostMesh> sampleMeshes(sampleSetSize);
	for(int i = 0; i < sampleSetSize; i++) {
		sampleMeshes.at(i) = hostLoadOBJ(filePaths.at(i));
	}

	// 4 Scale all models to fit in a 1x1x1 sphere
	std::cout << "Scaling meshes.." << std::endl;
	std::vector<HostMesh> scaledMeshes(sampleSetSize);
	for(int i = 0; i < sampleSetSize; i++) {
		scaledMeshes.at(i) = scaleMesh(sampleMeshes.at(i), 1);
	}

    // 5 Copy meshes to GPU
	std::vector<DeviceMesh> scaledMeshesOnGPU(sampleSetSize);
	for(int i = 0; i < sampleSetSize; i++) {
	    scaledMeshesOnGPU.at(i) = copyMeshToGPU(scaledMeshes.at(i));
	}

	// 6 Compute (quasi) spin images for all models in the sample set
	std::cout << "Computing reference QSI images.." << std::endl;
	std::vector<array<unsigned int>> originalDescriptors(sampleSetSize);
	for(int i = 0; i < sampleSetSize; i++) {
		std::cout << "\tComputing " << (i+1) << "/" << sampleSetSize << std::endl;
        generateQuasiSpinImages(scaledMeshesOnGPU.at(i), device_information, 3.0f);
	}

	// 7 Create a box of SxSxS units
	std::cout << "Creating a scene.." << std::endl;
	unsigned int totalVertexCount = 0;
	unsigned int totalIndexCount = 0;
	for(int i = 0; i < sampleSetSize; i++) {
		totalVertexCount += scaledMeshes.at(i).vertexCount;
		totalIndexCount += scaledMeshes.at(i).indexCount;
	}
	std::vector<unsigned int> box(boxSize);
	HostMesh boxScene(totalVertexCount, totalIndexCount);

	// 8 for all combinations (non-reused) models:
	std::cout << "Performing experiment                                                                                                                                                                                                                                              .." << std::endl;
	std::uniform_int_distribution<unsigned int> boxDistribution(0, sampleSetSize);
	unsigned int sceneVertexPointer = 0;
	unsigned int sceneIndexPointer = 0;
	for(unsigned int i = 0; i < sampleSetSize; i++) {
		// Subtracting i ensures the chosen index ranges between 0 and the number of remaining unselected samples
		unsigned int chosenIndex = boxDistribution(generator) - i;

		// 7.1 Copy
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