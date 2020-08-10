#include "quicciDistanceFunctionBenchmark.h"
#include "clutterSphereMeshAugmenter.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/mesh/MeshScaler.h>
#include <shapeDescriptor/utilities/mesh/OBJLoader.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/utilities/kernels/duplicateRemoval.cuh>
#include <utilities/randomFileSelector.h>
#include <cuda_runtime_api.h>
#include <nvidia/helper_cuda.h>
#include <shapeDescriptor/utilities/dumpers/meshDumper.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>
#include <cassert>
#include <utilities/stringUtils.h>
#include <json.hpp>
#include <tsl/ordered_map.h>
#include <shapeDescriptor/utilities/copy/mesh.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

void runQuicciDistanceFunctionBenchmark(
        std::experimental::filesystem::path sourceDirectory,
        std::experimental::filesystem::path outputDirectory,
        size_t seed,
        std::vector<int> sphereCountList,
        int sceneSphereCount,
        float clutterSphereRadius,
        GPUMetaData gpuMetaData,
        float supportRadius,
        BenchmarkMode mode) {
    // 1 Seeding the random number generator
    std::random_device rd("/dev/urandom");
    size_t randomSeed = seed != 0 ? seed : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::minstd_rand0 generator{randomSeed};

    std::cout << std::endl << "Running experiment initialisation sequence.." << std::endl;

    // 2 Search SHREC directory for files
    // 3 Make a sample set of n sample objects
    unsigned int sampleSetSize = 0;
    if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        sampleSetSize = 1;
    } else if(mode == BenchmarkMode::BASELINE) {
        sampleSetSize = 2;
    }
    std::vector<std::string> filePaths = generateRandomFileList(sourceDirectory.string(), sampleSetSize, generator);

    // 4 Load the models in the sample set
    std::cout << "\tLoading sample model.." << std::endl;
    ShapeDescriptor::cpu::Mesh sampleMesh = ShapeDescriptor::utilities::loadOBJ(filePaths.at(0), true);
    ShapeDescriptor::cpu::Mesh otherSampleMesh;
    if(mode == BenchmarkMode::BASELINE) {
        otherSampleMesh = ShapeDescriptor::utilities::loadOBJ(filePaths.at(1), true);
    }

    // 5 Scale all models to fit in a 1x1x1 sphere
    std::cout << "\tScaling meshes.." << std::endl;
    ShapeDescriptor::cpu::Mesh scaledMesh = ShapeDescriptor::utilities::fitMeshInsideSphereOfRadius(sampleMesh, 1);
    ShapeDescriptor::cpu::freeMesh(sampleMesh);
    ShapeDescriptor::cpu::Mesh scaledOtherSampleMesh;
    if(mode == BenchmarkMode::BASELINE) {
        scaledOtherSampleMesh = ShapeDescriptor::utilities::fitMeshInsideSphereOfRadius(otherSampleMesh, 1);
        ShapeDescriptor::cpu::freeMesh(otherSampleMesh);
    }

    // 6 Add clutter spheres to the mesh
    std::cout << "\tAugmenting mesh with spheres.." << std::endl;
    ShapeDescriptor::cpu::Mesh augmentedHostMesh;
    if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        augmentedHostMesh = applyClutterSpheres(scaledMesh, sceneSphereCount, clutterSphereRadius, generator());
        //ShapeDescriptor::dump::mesh(augmentedHostMesh, "dumped_sphere_mesh.obj");
    }

    // 6 Copy meshes to GPU
    std::cout << "\tCopying meshes to device.." << std::endl;
    ShapeDescriptor::gpu::Mesh unmodifiedMesh = ShapeDescriptor::copy::hostMeshToDevice(scaledMesh);
    ShapeDescriptor::cpu::freeMesh(scaledMesh);
    ShapeDescriptor::gpu::Mesh augmentedMesh;
    if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        augmentedMesh = ShapeDescriptor::copy::hostMeshToDevice(augmentedHostMesh);
    }
    ShapeDescriptor::gpu::Mesh otherSampleUnmodifiedMesh;
    if(mode == BenchmarkMode::BASELINE) {
        otherSampleUnmodifiedMesh = ShapeDescriptor::copy::hostMeshToDevice(scaledOtherSampleMesh);
        ShapeDescriptor::cpu::freeMesh(scaledOtherSampleMesh);
    }

    // 8 Remove duplicate vertices
    std::cout << "\tRemoving duplicate vertices.." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> imageOrigins = ShapeDescriptor::utilities::computeUniqueVertices(unmodifiedMesh);
    size_t imageCount = imageOrigins.length;
    size_t referenceImageCount = imageCount;
    size_t sampleImageCount = 0;
    std::cout << "\t\tReduced " << unmodifiedMesh.vertexCount << " vertices to " << imageCount << "." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> baselineOrigins;
    if(mode == BenchmarkMode::BASELINE) {
        baselineOrigins = ShapeDescriptor::utilities::computeUniqueVertices(otherSampleUnmodifiedMesh);
        sampleImageCount = baselineOrigins.length;
    }

    std::cout << "\tGenerating reference QUICCI images.." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_unmodifiedQuiccImages = ShapeDescriptor::gpu::generateQUICCImages(
            unmodifiedMesh,
            imageOrigins,
            supportRadius);

    size_t unmodifiedVertexCount = unmodifiedMesh.vertexCount;
    const size_t verticesPerSphere = SPHERE_VERTEX_COUNT;

    std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDistances>> measuredDistances;

    if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        for (unsigned int sphereCountIndex = 0; sphereCountIndex < sphereCountList.size(); sphereCountIndex++) {
            unsigned int sphereCount = sphereCountList.at(sphereCountIndex);
            augmentedMesh.vertexCount = unmodifiedVertexCount + sphereCount * verticesPerSphere;
            assert(sphereCount <= sceneSphereCount);
            std::cout << "\tComputing distances for a scene with " << sphereCount << " spheres.." << std::endl;
            std::cout << "\t\tGenerating QUICCI images.." << std::endl;
            ShapeDescriptor::debug::QUICCIExecutionTimes runInfo;
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_augmentedQuiccImages = ShapeDescriptor::gpu::generateQUICCImages(
                    augmentedMesh,
                    imageOrigins,
                    supportRadius,
                    &runInfo);

            std::cout << "\t\t\tTook " << runInfo.totalExecutionTimeSeconds << " seconds." << std::endl;

            std::cout << "\t\tComputing QUICCI distances.." << std::endl;
            ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDistances> sampleDistances = ShapeDescriptor::gpu::computeQUICCIElementWiseDistances(
                    device_unmodifiedQuiccImages,
                    device_augmentedQuiccImages);

            cudaFree(device_augmentedQuiccImages.content);

            measuredDistances.push_back(sampleDistances);
        }
    } else if(mode == BenchmarkMode::BASELINE) {
        imageCount = std::min<unsigned int>(imageCount, baselineOrigins.length);
        baselineOrigins.length = imageCount;

        std::cout << "\tComputing distances.." << std::endl;
        std::cout << "\t\tGenerating QUICCI images.." << std::endl;
        ShapeDescriptor::debug::QUICCIExecutionTimes runInfo;
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_sampleImages = ShapeDescriptor::gpu::generateQUICCImages(
                otherSampleUnmodifiedMesh,
                baselineOrigins,
                supportRadius,
                &runInfo);

        std::cout << "\t\t\tTook " << runInfo.totalExecutionTimeSeconds << " seconds." << std::endl;

        std::cout << "\t\tComputing QUICCI distances.." << std::endl;
        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDistances> sampleDistances = ShapeDescriptor::gpu::computeQUICCIElementWiseDistances(
                device_unmodifiedQuiccImages,
                device_sampleImages);

        cudaFree(device_sampleImages.content);

        measuredDistances.push_back(sampleDistances);
    }

    std::cout << "Experiments done, dumping results.." << std::endl;

    std::string timestring = getCurrentDateTimeString();
    std::string outputFileName = "quicciDistances_" + timestring + "_" + std::to_string(randomSeed) + ".json";
    std::experimental::filesystem::path outputFilePath =  outputDirectory / outputFileName;

    json outJson;

    outJson["version"] = "v2";
    outJson["seed"] = seed;
    outJson["spinImageWidthPixels"] = spinImageWidthPixels;
    outJson["sphereCounts"] = sphereCountList;
    outJson["clutterSphereRadius"] = clutterSphereRadius;
    outJson["sceneSphereCount"] = sceneSphereCount;
    outJson["sphereSliceCount"] = SPHERE_RESOLUTION_X;
    outJson["sphereLayerCount"] = SPHERE_RESOLUTION_Y;
    outJson["trianglesPerSphere"] = SPHERE_TRIANGLE_COUNT;
    outJson["verticesPerSphere"] = SPHERE_VERTEX_COUNT;
    outJson["supportRadius"] = supportRadius;
    outJson["chosenObjectPath"] = filePaths.at(0);
    outJson["imageCount"] = imageCount;
    outJson["objectVertexCount"] = unmodifiedVertexCount;
    outJson["gpuInfo"] = {};
    outJson["gpuInfo"]["name"] = gpuMetaData.name;
    outJson["gpuInfo"]["clockrate"] = gpuMetaData.clockRate;
    outJson["gpuInfo"]["memoryCapacityInMB"] = gpuMetaData.memorySizeMB;
    if(mode == BenchmarkMode::BASELINE) {
        outJson["mode"] = "baseline";
        outJson["primaryObjectImageCount"] = referenceImageCount;
        outJson["secondaryObjectImageCount"] = sampleImageCount;
        outJson["secondaryObjectPath"] = filePaths.at(1);
    } else if(mode == BenchmarkMode::SPHERE_CLUTTER) {
        outJson["mode"] = "similar";
    }

    outJson["measuredDistances"] = {};
    outJson["measuredDistances"]["clutterResistant"] = {};
    outJson["measuredDistances"]["weightedHamming"] = {};
    outJson["measuredDistances"]["hamming"] = {};
    outJson["imageBitCounts"] = {};
    for(unsigned int sphereCountIndex = 0; sphereCountIndex < sphereCountList.size(); sphereCountIndex++) {
        std::string keyName = std::to_string(sphereCountList.at(sphereCountIndex)) + " spheres";
        outJson["measuredDistances"]["clutterResistant"][keyName] = {};
        outJson["measuredDistances"]["weightedHamming"][keyName] = {};
        outJson["measuredDistances"]["hamming"][keyName] = {};

        for(int image = 0; image < imageCount; image++) {
            outJson["measuredDistances"]["clutterResistant"][keyName].push_back(
                    measuredDistances.at(sphereCountIndex).content[image].clutterResistantDistance);
            outJson["measuredDistances"]["weightedHamming"][keyName].push_back(
                    measuredDistances.at(sphereCountIndex).content[image].weightedHammingDistance);
            outJson["measuredDistances"]["hamming"][keyName].push_back(
                    measuredDistances.at(sphereCountIndex).content[image].hammingDistance);
            if(sphereCountIndex == 0) {
                outJson["imageBitCounts"].push_back(
                    measuredDistances.at(sphereCountIndex).content[image].needleImageBitCount);
            }
        }
    }

    std::ofstream outFile(outputFilePath);
    outFile << outJson.dump(4);
    outFile.close();

    std::cout << std::endl << "Complete." << std::endl;
}