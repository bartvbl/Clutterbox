#include "clutterBoxExperiment.hpp"

#include <cuda_runtime_api.h>

#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

#include <utilities/stringUtils.h>
#include <shapeDescriptor/utilities/mesh/MeshScaler.h>
#include <utilities/Histogram.h>

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/kernels/meshSampler.cuh>
#include <shapeDescriptor/utilities/free/mesh.h>

#include <clutterbox/clutterBoxUtilities.h>
#include <glm/vec3.hpp>
#include <map>
#include <json.hpp>
#include <tsl/ordered_map.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/utilities/dump/meshDumper.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <utilities/randomFileSelector.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

#include "clutterBoxKernels.cuh"

#include "utilities/listDir.h"
#include "nvidia/helper_cuda.h"

Histogram computeSearchResultHistogram(size_t vertexCount, const ShapeDescriptor::cpu::array<unsigned int> &searchResults);

void dumpResultsFile(
        std::string outputFile,
        std::vector<Clutterbox::Method*> descriptorsToEvaluate,
        size_t seed,
        std::vector<std::vector<Histogram>> histograms,
        std::string sourceDirectory,
        std::vector<int> objectCountList,
        int overrideObjectCount,
        float boxSize,
        float supportRadius,
        unsigned long assertionRandomToken,
        std::vector<ExecutionTimes>* referenceExecutionTimes,
        std::vector<std::vector<ExecutionTimes>>* sampleExecutionTimes,
        std::vector<std::vector<ExecutionTimes>>* searchExecutionTimes,
        std::vector<size_t> uniqueVertexCounts,
        std::vector<size_t> pointCloudSampleCounts,
        GPUMetaData gpuMetaData) {

    std::cout << std::endl << "Dumping results file to " << outputFile << ".." << std::endl;

    std::minstd_rand0 generator{seed};

    int sampleObjectCount = *std::max_element(objectCountList.begin(), objectCountList.end());

    if(overrideObjectCount != -1) {
        sampleObjectCount = overrideObjectCount;
    }

    std::vector<std::string> chosenFiles = generateRandomFileList(sourceDirectory, sampleObjectCount, generator);

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
    }

    std::vector<ShapeDescriptor::cpu::Mesh> sampleMeshes(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        sampleMeshes.at(i) = ShapeDescriptor::utilities::loadOBJ(chosenFiles.at(i), true);
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

    std::vector<std::string> descriptorNames;
    for(int i = 0; i < descriptorsToEvaluate.size(); i++) {
        descriptorNames.push_back(descriptorsToEvaluate.at(i)->getMethodCommandLineParameterName());
    }

    json outJson;

    outJson["version"] = "v15";
    outJson["seed"] = seed;
    outJson["descriptors"] = descriptorNames;
    outJson["sampleSetSize"] = sampleObjectCount;
    outJson["sampleObjectCounts"] = objectCountList;
    outJson["overrideObjectCount"] = overrideObjectCount;
    outJson["uniqueVertexCounts"] = uniqueVertexCounts;
    outJson["imageCounts"] = uniqueVertexCounts;
    outJson["boxSize"] = boxSize;
    outJson["supportRadius"] = supportRadius;
    outJson["pointCloudSampleCounts"] = pointCloudSampleCounts;
    outJson["searchResultCount"] = SEARCH_RESULT_COUNT;
    for(int descriptorIndex = 0; descriptorIndex < descriptorsToEvaluate.size(); descriptorIndex++) {
        descriptorsToEvaluate.at(descriptorIndex)->dumpMetadata(&outJson);
    }
    outJson["gpuInfo"] = {};
    outJson["gpuInfo"]["name"] = gpuMetaData.name;
    outJson["gpuInfo"]["clockrate"] = gpuMetaData.clockRate;
    outJson["gpuInfo"]["memoryCapacityInMB"] = gpuMetaData.memorySizeMB;
    outJson["inputFiles"] = chosenFiles;
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

    // Dump execution times
    for(int descriptorIndex = 0; descriptorIndex < descriptorsToEvaluate.size(); descriptorIndex++) {
        std::string descriptorName = descriptorsToEvaluate.at(descriptorIndex)->getMethodDumpFileName();
        for(const auto &i : *referenceExecutionTimes->at(descriptorIndex).getAll()) {
            outJson["runtimes"][descriptorName + "ReferenceGeneration"][i.name] = i.timeInSeconds;
        }
        for(ExecutionTimes &i : sampleExecutionTimes->at(descriptorIndex)) {
            for(const auto &j : *i.getAll()) {
                outJson["runtimes"][descriptorName + "SampleGeneration"][j.name].push_back(j.timeInSeconds);
            }
        }
        for(ExecutionTimes &i : searchExecutionTimes->at(descriptorIndex)) {
            for(const auto &j : *i.getAll()) {
                 outJson["runtimes"][descriptorName + "Search"][j.name].push_back(j.timeInSeconds);
            }
        }
    }

    // Dump histograms
    for(int descriptorIndex = 0; descriptorIndex < descriptorsToEvaluate.size(); descriptorIndex++) {
        std::string descriptorName = descriptorsToEvaluate.at(descriptorIndex)->getMethodDumpFileName();
        outJson[descriptorName + "histograms"] = {};
        for(unsigned int i = 0; i < objectCountList.size(); i++) {
            std::map<unsigned int, size_t> descriptorMap = histograms.at(descriptorIndex).at(i).getMap();
            std::vector<unsigned int> keys;
            for (auto &content : descriptorMap) {
                keys.push_back(content.first);
            }
            std::sort(keys.begin(), keys.end());

            for(auto &key : keys) {
                outJson[descriptorName + "histograms"][std::to_string(objectCountList.at(i)) + " objects"][std::to_string(key)] = descriptorMap[key];
            }
        }
    }

    std::ofstream outFile(outputFile);
    outFile << outJson.dump(4);
    outFile.close();

    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        ShapeDescriptor::free::mesh(sampleMeshes.at(i));
    }
}

void dumpRawSearchResultFile(
        std::string outputFile,
        std::vector<Clutterbox::Method*> descriptorsToEvaluate,
        std::vector<int> objectCountList,
        std::vector<std::vector<ShapeDescriptor::cpu::array<unsigned int>>> rawSearchResults,
        size_t seed) {

    std::cout << "Dumping raw search results to " << outputFile << ".." << std::endl;

    json outJson;

    outJson["version"] = "rawfile_v4";
    outJson["sampleObjectCounts"] = objectCountList;
    outJson["seed"] = seed;


    for(int descriptorIndex = 0; descriptorIndex < descriptorsToEvaluate.size(); descriptorIndex++) {
        std::string descriptorName = descriptorsToEvaluate.at(descriptorIndex)->getMethodDumpFileName();

        outJson[descriptorName] = {};
        for(int i = 0; i < rawSearchResults.at(descriptorIndex).size(); i++) {
            std::string indexString = std::to_string(objectCountList.at(i));
            outJson[descriptorName][indexString] = {};
            for(int j = 0; j < rawSearchResults.at(descriptorIndex).at(i).length; j++) {
                outJson[descriptorName][indexString].push_back(rawSearchResults.at(descriptorIndex).at(i).content[j]);
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

void dumpSearchResultVisualisationMesh(const ShapeDescriptor::cpu::array<unsigned int> &searchResults,
                                       const ShapeDescriptor::gpu::Mesh &referenceDeviceMesh,
                                       const std::experimental::filesystem::path outFilePath,
                                       unsigned int resultThreshold) {
    size_t totalUniqueVertexCount;
    std::vector<size_t> vertexCounts;
    ShapeDescriptor::gpu::array<signed long long> device_indexMapping = ShapeDescriptor::utilities::computeUniqueIndexMapping(referenceDeviceMesh, {referenceDeviceMesh}, &vertexCounts, totalUniqueVertexCount);

    ShapeDescriptor::cpu::Mesh hostMesh = ShapeDescriptor::copy::deviceMeshToHost(referenceDeviceMesh);

    size_t referenceMeshVertexCount = referenceDeviceMesh.vertexCount;
    ShapeDescriptor::cpu::array<signed long long> host_indexMapping = {0, nullptr};
    host_indexMapping.content = new signed long long[referenceMeshVertexCount];
    host_indexMapping.length = referenceMeshVertexCount;
    cudaMemcpy(host_indexMapping.content, device_indexMapping.content, referenceMeshVertexCount * sizeof(signed long long), cudaMemcpyDeviceToHost);
    cudaFree(device_indexMapping.content);

    ShapeDescriptor::cpu::array<float2> textureCoords = {referenceMeshVertexCount, new float2[referenceMeshVertexCount]};
    for(size_t vertexIndex = 0; vertexIndex < referenceMeshVertexCount; vertexIndex++) {

        ShapeDescriptor::cpu::float3 vertex = hostMesh.vertices[vertexIndex];
        ShapeDescriptor::cpu::float3 normal = hostMesh.normals[vertexIndex];
        size_t targetIndex = 0;
        for(size_t duplicateVertexIndex = 0; duplicateVertexIndex < hostMesh.vertexCount; duplicateVertexIndex++) {
            ShapeDescriptor::cpu::float3 otherVertex = hostMesh.vertices[duplicateVertexIndex];
            ShapeDescriptor::cpu::float3 otherNormal = hostMesh.normals[duplicateVertexIndex];
            if(vertex == otherVertex && normal == otherNormal) {
                break;
            }
            if(host_indexMapping.content[duplicateVertexIndex] != -1) {
                targetIndex++;
            }
        }
        unsigned int searchResult = searchResults.content[targetIndex];

        // Entry has been marked as duplicate
        // So we need to find the correct index

        float texComponent = searchResult == resultThreshold ? 0.5 : 0;
        float2 texCoord = {texComponent, texComponent};
        textureCoords.content[vertexIndex] = texCoord;
    }

    ShapeDescriptor::dump::mesh(hostMesh, outFilePath, textureCoords, "colourTexture.png");

    delete[] host_indexMapping.content;
    ShapeDescriptor::free::mesh(hostMesh);
}

void runClutterBoxExperiment(
        std::string objectDirectory,
        std::vector<Clutterbox::Method*> descriptorsToEvaluate,
        std::vector<int> objectCountList,
        int overrideObjectCount,
        float boxSize,
        float supportRadius,
        bool dumpRawSearchResults,
        std::string outputDirectory,
        bool dumpSceneOBJFiles,
        std::string sceneOBJFileDumpDir,
        bool enableMatchVisualisation,
        std::string matchVisualisationOutputDir,
        std::vector<std::string> matchVisualisationDescriptorList,
        unsigned int matchVisualisationThreshold,
        GPUMetaData gpuMetaData,
        size_t overrideSeed) {

    std::cout << "Running clutterbox experiment for the following descriptors: ";

    for(int i = 0; i < descriptorsToEvaluate.size(); i++) {
        std::cout << descriptorsToEvaluate.at(i)->getMethodCommandLineParameterName();
        if(i + 1 < descriptorsToEvaluate.size()) {
            std::cout << ", ";
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

    std::vector<std::vector<Histogram>> histograms;
    histograms.resize(descriptorsToEvaluate.size());

    std::vector<ExecutionTimes> generationReferenceExecutionTimes;
    generationReferenceExecutionTimes.resize(descriptorsToEvaluate.size());

    std::vector<std::vector<ExecutionTimes>> generationSampleExecutionTimes;
    generationSampleExecutionTimes.resize(descriptorsToEvaluate.size());

    std::vector<std::vector<ExecutionTimes>> searchExecutionTimes;
    searchExecutionTimes.resize(descriptorsToEvaluate.size());

    std::vector<std::vector<ShapeDescriptor::cpu::array<unsigned int>>> rawSearchResults;
    rawSearchResults.resize(descriptorsToEvaluate.size());

    std::vector<size_t> spinImageSampleCounts;

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
    std::random_device rd("/dev/urandom");
    size_t randomSeed = overrideSeed != 0 ? overrideSeed : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::minstd_rand0 generator{randomSeed};

    std::cout << std::endl << "Running experiment initialisation sequence.." << std::endl;

    // 2 Search SHREC directory for files
    // 3 Make a sample set of n sample objects
    std::vector<std::string> filePaths = generateRandomFileList(objectDirectory, sampleObjectCount, generator);

    // 4 Load the models in the sample set
    std::cout << "\tLoading sample models.." << std::endl;
    std::vector<ShapeDescriptor::cpu::Mesh> sampleMeshes(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        sampleMeshes.at(i) = ShapeDescriptor::utilities::loadOBJ(filePaths.at(i), true);
        std::cout << "\t\tMesh " << i << ": " << sampleMeshes.at(i).vertexCount << " vertices" << std::endl;
    }

    // 5 Scale all models to fit in a 1x1x1 sphere
    std::cout << "\tScaling meshes.." << std::endl;
    std::vector<ShapeDescriptor::cpu::Mesh> scaledMeshes(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        scaledMeshes.at(i) = ShapeDescriptor::utilities::fitMeshInsideSphereOfRadius(sampleMeshes.at(i), 1);
        ShapeDescriptor::free::mesh(sampleMeshes.at(i));
    }

    // 6 Copy meshes to GPU
    std::cout << "\tCopying meshes to device.." << std::endl;
    std::vector<ShapeDescriptor::gpu::Mesh> scaledMeshesOnGPU(sampleObjectCount);
    for (unsigned int i = 0; i < sampleObjectCount; i++) {
        scaledMeshesOnGPU.at(i) = ShapeDescriptor::copy::hostMeshToDevice(scaledMeshes.at(i));
    }

    // 7 Shuffle the list. First mesh is now our "reference".
    std::cout << "\tShuffling sample object list.." << std::endl;
    std::shuffle(std::begin(scaledMeshesOnGPU), std::end(scaledMeshesOnGPU), generator);

    // 8 Remove duplicate vertices
    std::cout << "\tRemoving duplicate vertices.." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> spinOrigins_reference = ShapeDescriptor::utilities::computeUniqueVertices(scaledMeshesOnGPU.at(0));
    size_t referenceImageCount = spinOrigins_reference.length;
    std::cout << "\t\tReduced " << scaledMeshesOnGPU.at(0).vertexCount << " vertices to " << referenceImageCount << "." << std::endl;

    const size_t referenceSampleCount = computeSpinImageSampleCount(scaledMeshesOnGPU.at(0).vertexCount);
    std::cout << "\tUsing sample count: " << referenceSampleCount << std::endl;

    // 9 Compute descriptors for reference model
    std::cout << "Generating reference descriptors.." << std::endl;

    std::vector<ShapeDescriptor::gpu::array<char>> referenceDescriptors;

    size_t referenceGenerationRandomSeed = generator();
    ShapeDescriptor::gpu::PointCloud device_referencePointCloud =
            ShapeDescriptor::utilities::sampleMesh(scaledMeshesOnGPU.at(0), referenceSampleCount, referenceGenerationRandomSeed);

    for(int i = 0; i < descriptorsToEvaluate.size(); i++) {
        std::cout << "\tGenerating reference " + descriptorsToEvaluate.at(i)->getMethodDumpFileName() + " descriptors.." << std::endl;

        Clutterbox::GenerationParameters parameters;
        parameters.supportRadius = supportRadius;

        ExecutionTimes executionTimes;

        referenceDescriptors.push_back(descriptorsToEvaluate.at(i)->generateDescriptors(
                scaledMeshesOnGPU.at(0),
                device_referencePointCloud,
                spinOrigins_reference,
                parameters,
                &executionTimes));

        generationReferenceExecutionTimes.at(i) = executionTimes;

        std::cout << "\t\tExecution time: " << executionTimes.getMeasurementByName("total") << "s" << std::endl;
    }

    device_referencePointCloud.free();

    checkCudaErrors(cudaFree(spinOrigins_reference.content));

    // 10 Combine meshes into one larger scene
    ShapeDescriptor::gpu::Mesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

    // 11 Compute unique vertex mapping
    std::vector<size_t> uniqueVertexCounts;
    size_t totalUniqueVertexCount;
    ShapeDescriptor::gpu::array<signed long long> device_indexMapping = ShapeDescriptor::utilities::computeUniqueIndexMapping(boxScene, scaledMeshesOnGPU, &uniqueVertexCounts, totalUniqueVertexCount);

    // 12 Randomly transform objects
    std::cout << "\tRandomly transforming input objects.." << std::endl;
    randomlyTransformMeshes(boxScene, boxSize, scaledMeshesOnGPU, generator);

    size_t vertexCount = 0;
    size_t referenceMeshImageCount = spinOrigins_reference.length;

    // 13 Compute corresponding transformed vertex buffer
    //    A mapping is used here because the previously applied transformation can cause non-unique vertices to become
    //    equivalent. It is vital we can rely on a 1:1 mapping existing between vertices.
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_uniqueSpinOrigins = ShapeDescriptor::utilities::applyUniqueMapping(boxScene, device_indexMapping, totalUniqueVertexCount);
    checkCudaErrors(cudaFree(device_indexMapping.content));
    size_t imageCount = 0;

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

        // Do mesh sampling
        std::cout << "\t" << std::endl;
        size_t meshSamplingSeed = generator();
        size_t pointCloudSampleCount = computeSpinImageSampleCount(imageCount);
        spinImageSampleCounts.push_back(pointCloudSampleCount);

        // wasteful solution, but I don't want to do ugly hacks that destroy the function APIs
        // Computes number of samples used for the reference object
        ShapeDescriptor::internal::MeshSamplingBuffers sampleBuffers;
        ShapeDescriptor::gpu::PointCloud device_pointCloud = ShapeDescriptor::utilities::sampleMesh(boxScene, pointCloudSampleCount, meshSamplingSeed, &sampleBuffers);
        float totalArea;
        float referenceObjectTotalArea;
        size_t referenceObjectTriangleCount = scaledMeshesOnGPU.at(0).vertexCount / 3;
        size_t sceneTriangleCount = boxScene.vertexCount / 3;
        checkCudaErrors(cudaMemcpy(&totalArea,
                                   sampleBuffers.cumulativeAreaArray.content + (sceneTriangleCount - 1),
                                   sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&referenceObjectTotalArea,
                                   sampleBuffers.cumulativeAreaArray.content + (referenceObjectTriangleCount - 1),
                                   sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(sampleBuffers.cumulativeAreaArray.content);
        float areaFraction = referenceObjectTotalArea / totalArea;
        size_t currentReferenceObjectSampleCount = size_t(double(areaFraction) * double(pointCloudSampleCount));
        std::cout << "\t\tReference object sample count: " << currentReferenceObjectSampleCount << std::endl;


        for(int methodIndex = 0; methodIndex < descriptorsToEvaluate.size(); methodIndex++) {
            std::cout << "\tGenerating " + descriptorsToEvaluate.at(methodIndex)->getMethodDumpFileName()
                       + " descriptors.. (" << imageCount << " descriptors)" << std::endl;

            ExecutionTimes generationExecutionTimes;

            Clutterbox::GenerationParameters parameters;
            parameters.supportRadius = supportRadius;

            ShapeDescriptor::gpu::array<char> sampleDescriptors = descriptorsToEvaluate.at(methodIndex)->generateDescriptors(boxScene, device_pointCloud, device_uniqueSpinOrigins, parameters, &generationExecutionTimes);

            generationSampleExecutionTimes.at(methodIndex).push_back(generationExecutionTimes);

            std::cout << "\t\tExecution time: " << generationExecutionTimes.getMeasurementByName("total") << "s" << std::endl;

            std::cout << "\tSearching in " + descriptorsToEvaluate.at(methodIndex)->getMethodDumpFileName()
                       + " descriptors.." << std::endl;

            ExecutionTimes sampleSearchExecutionTimes;

            Clutterbox::SearchParameters searchParameters;
            searchParameters.haystackDescriptorScenePointCloudPointCount = pointCloudSampleCount;
            searchParameters.needleDescriptorScenePointCloudPointCount = referenceSampleCount;

            ShapeDescriptor::gpu::array<char> methodReferenceDescriptors = referenceDescriptors.at(methodIndex);
            ShapeDescriptor::cpu::array<unsigned int> searchResults = descriptorsToEvaluate.at(methodIndex)->computeSearchResultRanks(methodReferenceDescriptors, sampleDescriptors, searchParameters, &sampleSearchExecutionTimes);

            searchExecutionTimes.at(methodIndex).push_back(sampleSearchExecutionTimes);

            rawSearchResults.at(methodIndex).push_back(searchResults);

            std::cout << "\t\tExecution time: " << sampleSearchExecutionTimes.getMeasurementByName("total") << "s" << std::endl;
            Histogram histogram = computeSearchResultHistogram(referenceMeshImageCount, searchResults);

            if(enableMatchVisualisation && std::find(matchVisualisationDescriptorList.begin(), matchVisualisationDescriptorList.end(), descriptorsToEvaluate.at(methodIndex)->getMethodCommandLineParameterName()) != matchVisualisationDescriptorList.end()) {
                std::cout << "\tDumping OBJ visualisation of search results.." << std::endl;
                std::experimental::filesystem::path outFilePath = matchVisualisationOutputDir;
                outFilePath = outFilePath / ("searchresults_" + std::to_string(randomSeed)
                                  + "_" + descriptorsToEvaluate.at(methodIndex)->getMethodCommandLineParameterName()
                                  + "_" + std::to_string(objectCount + 1) + ".obj");
                dumpSearchResultVisualisationMesh(searchResults, scaledMeshesOnGPU.at(0), outFilePath, matchVisualisationThreshold);
            }

            if(!dumpRawSearchResults) {
                delete[] searchResults.content;
            }

            // Storing results
            histograms.at(methodIndex).push_back(histogram);

            // Finally, delete the descriptors
            cudaFree(sampleDescriptors.content);
        }

        // Dumping OBJ file of current scene, if enabled
        if(dumpSceneOBJFiles) {
            ShapeDescriptor::cpu::Mesh hostMesh = ShapeDescriptor::copy::deviceMeshToHost(boxScene);

            std::experimental::filesystem::path outFilePath = sceneOBJFileDumpDir;
            outFilePath = outFilePath / ("scene_" + std::to_string(randomSeed) + "_" + std::to_string(objectCount + 1) + ".obj");

            std::cout << "\tDumping OBJ file of scene to " << outFilePath << std::endl;

            ShapeDescriptor::dump::mesh(hostMesh, outFilePath, 0, scaledMeshesOnGPU.at(0).vertexCount);

            ShapeDescriptor::free::mesh(hostMesh);
        }
    }

    // Cleaning up
    ShapeDescriptor::gpu::freeMesh(boxScene);
    for(int i = 0; i < descriptorsToEvaluate.size(); i++) {
        cudaFree(referenceDescriptors.at(i).content);
    }

    std::string timestring = getCurrentDateTimeString();

    dumpResultsFile(
            outputDirectory + timestring + "_" + std::to_string(randomSeed) + ".json",
            descriptorsToEvaluate,
            randomSeed,
            histograms,
            objectDirectory,
            objectCountList,
            overrideObjectCount,
            boxSize,
            supportRadius,
            generator(),
            &generationReferenceExecutionTimes,
            &generationSampleExecutionTimes,
            &searchExecutionTimes,
            uniqueVertexCounts,
            spinImageSampleCounts,
            gpuMetaData);

    if(dumpRawSearchResults) {
        dumpRawSearchResultFile(
                outputDirectory + "raw/" + timestring + "_" + std::to_string(randomSeed) + ".json",
                descriptorsToEvaluate,
                objectCountList,
                rawSearchResults,
                randomSeed);

        // Cleanup
        // If one of the descriptors is not enabled, this will iterate over an empty vector.
        for(int i = 0; i < descriptorsToEvaluate.size(); i++) {
            for(auto results : rawSearchResults.at(i)) {
                delete[] results.content;
            }
        }
    }

    for(ShapeDescriptor::gpu::Mesh deviceMesh : scaledMeshesOnGPU) {
        ShapeDescriptor::gpu::freeMesh(deviceMesh);
    }

    std::cout << std::endl;
    std::cout << "Experiment complete. Results have been written to:" << std::endl;
    std::cout << std::endl << "    " << outputDirectory + timestring + "_" + std::to_string(randomSeed) + ".json" << std::endl;
    std::cout << std::endl;
}



Histogram computeSearchResultHistogram(size_t vertexCount, const ShapeDescriptor::cpu::array<unsigned int> &searchResults) {

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
