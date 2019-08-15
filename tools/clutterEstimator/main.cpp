#include <fstream>
#include <string>
#include <random>

#include <arrrgh.hpp>
#include <json.hpp>
#include <cuda_runtime_api.h>

#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <experiments/clutterBox/clutterBoxUtilities.h>
#include <spinImage/gpu/types/GPUPointCloud.h>
#include <spinImage/utilities/meshSampler.cuh>
#include <utilities/stringUtils.h>

using json = nlohmann::json;

#include "experimentUtilities/listDir.h"
#include "experiments/clutterBox/clutterBoxKernels.cuh"
#include "utilities/modelScaler.h"
#include "clutterKernel.cuh"
#include "../../../libShapeSearch/lib/nvidia-samples-common/nvidia/helper_cuda.h"

void stringSplit(std::vector<std::string>* parts, const std::string &s, char delim) {

    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        parts->push_back(item);
    }
}

int main(int argc, const char** argv) {

    const std::string DIRECTORY_UNSPECIFIED = "UNDEFINED";

    arrrgh::parser parser("clutterEstimator", "Estimates amount of clutter for each experiment iteration");
    const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& resultDir = parser.add<std::string>("result-dump-dir", "Define the directory containing experiment output dumps.", '\0', arrrgh::Optional, DIRECTORY_UNSPECIFIED);
    const auto& objectDir = parser.add<std::string>("object-dir", "Define the directory containing input objects.", '\0', arrrgh::Optional, DIRECTORY_UNSPECIFIED);
    const auto& outDir = parser.add<std::string>("output-dir", "Define the directory where computed clutter values should be dumped.", '\0', arrrgh::Optional, DIRECTORY_UNSPECIFIED);


    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    if(resultDir.value() == DIRECTORY_UNSPECIFIED) {
        std::cout << "A result directory must be specified!" << std::endl;
        return 0;
    }

    if(objectDir.value() == DIRECTORY_UNSPECIFIED) {
        std::cout << "An object input directory must be specified!" << std::endl;
        return 0;
    }

    if(outDir.value() == DIRECTORY_UNSPECIFIED) {
        std::cout << "An output directory must be specified!" << std::endl;
        return 0;
    }

    std::cout << "Listing object directory..";
    std::vector<std::string> objectFileList = listDir(objectDir.value());
    std::cout << " (found " << objectFileList.size() << " files)" << std::endl;

    std::cout << "Listing result directory..";
    std::vector<std::string> resultFileList = listDir(resultDir.value());
    std::cout << " (found " << resultFileList.size() << " files)" << std::endl;
    std::cout << std::endl;


    std::vector<std::string> parts;

    for (unsigned int resultFileIndex = 0; resultFileIndex < resultFileList.size(); resultFileIndex++) {
        std::string resultFile = resultFileList.at(resultFileIndex);
        if(resultFile == "raw") {
            continue;
        }
        std::cout << "Processing " << (resultFileIndex + 1) << "/" << resultFileList.size() << ": " << resultFile << std::endl;
        std::ifstream inputResultFile(resultDir.value() + "/" + resultFile);
        json resultFileContents;
        inputResultFile >> resultFileContents;
        inputResultFile.close();

        int sampleObjectCount = resultFileContents["sampleSetSize"];
        float boxSize = float(resultFileContents["boxSize"]);

        std::vector<HostMesh> objects;
        objects.resize(sampleObjectCount);

        for(unsigned int object = 0; object < sampleObjectCount; object++) {
            std::string objectFile = resultFileContents["inputFiles"].at(object);
            stringSplit(&parts, objectFile, '/');
            std::cout << "\r\t" << "Loading object " << (object + 1) << "/" << sampleObjectCount << ": " << parts.at(parts.size() - 1) << std::flush;
            objects.at(object) = SpinImage::utilities::loadOBJ(objectDir.value() + "/" + parts.at(parts.size() - 1));
        }
        std::cout << std::endl;

        // 1 Seeding the random number generator
        std::random_device rd;
        size_t randomSeed = resultFileContents["seed"];
        std::cout << "\tRandom seed: " << randomSeed << std::endl;
        std::default_random_engine generator{randomSeed};

        // This simulates the random generator calls for picking the random objects
        // Used for consistent RNG when translating/rotating objects
        std::shuffle(std::begin(objectFileList), std::end(objectFileList), generator);
        std::vector<HostMesh> tempVector;
        tempVector.resize(sampleObjectCount);
        std::shuffle(std::begin(tempVector), std::end(tempVector), generator);
        // This represents an random number generation for the spin image seed selection
        generator();

        // 5 Scale all models to fit in a 1x1x1 sphere
        std::cout << "\tScaling meshes.." << std::endl;
        std::vector<HostMesh> scaledMeshes(sampleObjectCount);
        for (unsigned int i = 0; i < sampleObjectCount; i++) {
            scaledMeshes.at(i) = fitMeshInsideSphereOfRadius(objects.at(i), 1);
            SpinImage::cpu::freeHostMesh(objects.at(i));
        }

        // 6 Copy meshes to GPU
        std::cout << "\tCopying meshes to device.." << std::endl;
        std::vector<DeviceMesh> scaledMeshesOnGPU(sampleObjectCount);
        for (unsigned int i = 0; i < sampleObjectCount; i++) {
            scaledMeshesOnGPU.at(i) = SpinImage::copy::hostMeshToDevice(scaledMeshes.at(i));
        }

        // 10 Combine meshes into one larger scene
        DeviceMesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

        // 11 Compute unique vertex mapping
        std::vector<size_t> uniqueVertexCounts;
        size_t totalUniqueVertexCount;
        array<signed long long> device_indexMapping = computeUniqueIndexMapping(boxScene, scaledMeshesOnGPU, &uniqueVertexCounts, totalUniqueVertexCount);

        // 12 Randomly transform objects
        std::cout << "\tRandomly transforming input objects.." << std::endl;
        randomlyTransformMeshes(boxScene, boxSize, scaledMeshesOnGPU, generator);

        // 13 Compute corresponding transformed vertex buffer
        //    A mapping is used here because the previously applied transformation can cause non-unique vertices to become
        //    equivalent. It is vital we can rely on a 1:1 mapping existing between vertices.
        array<DeviceOrientedPoint> device_uniqueSpinOrigins = applyUniqueMapping(boxScene, device_indexMapping, totalUniqueVertexCount);
        checkCudaErrors(cudaFree(device_indexMapping.content));

        size_t sampleCount = std::max(10 * boxScene.vertexCount, (size_t) 1000000);


        std::cout << "\tSampling scene.. (" << sampleCount << " samples)" << std::endl;
        SpinImage::internal::MeshSamplingBuffers sampleBuffers;
        SpinImage::GPUPointCloud sampledScene = SpinImage::utilities::sampleMesh(boxScene, sampleCount, generator(), &sampleBuffers);

        std::cout << "\tComputing reference object sample count.." << std::endl;
        size_t referenceObjectSampleCount = computeReferenceSampleCount(scaledMeshesOnGPU.at(0), sampleCount, sampleBuffers.cumulativeAreaArray);
        std::cout << "\t\tReference object has " << referenceObjectSampleCount << " samples." << std::endl;

        std::cout << "\tComputing clutter values.." << std::endl;
        float spinImageWidth = resultFileContents["spinImageWidth"];

        array<float> clutterValues = computeClutter(device_uniqueSpinOrigins, sampledScene, spinImageWidth, referenceObjectSampleCount, resultFileContents["uniqueVertexCounts"][0]);

        json outJson;

        outJson["version"] = "clutter_v1";
        outJson["clutterValues"] = {};
        outJson["sourceFile"] = resultFile;

        for(size_t item = 0; item < clutterValues.length; item++) {
            outJson["clutterValues"].push_back(clutterValues.content[item]);
        }

        std::cout << "Writing output file.." << std::endl;
        std::string outFilePath = outDir.value() + "/" + getCurrentDateTimeString() + ".json";
        std::ofstream outFile(outFilePath);
        outFile << outJson.dump(4) << std::endl;
        outFile.close();

        std::cout << "Cleaning up.." << std::endl;
        sampledScene.free();
        cudaFree(device_uniqueSpinOrigins.content);
        SpinImage::gpu::freeDeviceMesh(boxScene);

        for(DeviceMesh deviceMesh : scaledMeshesOnGPU) {
            SpinImage::gpu::freeDeviceMesh(deviceMesh);
        }

        std::cout << "Done." << std::endl;
    }

}