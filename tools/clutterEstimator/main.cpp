#include <string>
#include <random>

#include <arrrgh.hpp>
#include <json.hpp>
#include <cuda_runtime_api.h>

#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <clutterbox/clutterBoxUtilities.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/utilities/kernels/meshSampler.cuh>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/mesh/MeshScaler.h>
#include <utilities/stringUtils.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/free/mesh.h>

using json = nlohmann::json;

#include "utilities/listDir.h"
#include "clutterbox/clutterBoxKernels.cuh"
#include "clutterKernel.cuh"
#include "nvidia/helper_cuda.h"

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
    const auto& startIndex = parser.add<int>("start-index", "Start processing at the given file index.", '\0', arrrgh::Optional, 0);
    const auto& computeSingleIndex = parser.add<int>("compute-single-index", "Instead of processing the entire directory, only process one clutter file.", '\0', arrrgh::Optional, -1);
    const auto& forceGPU = parser.add<int>("force-gpu", "Force using the GPU with the given ID", '\0', arrrgh::Optional, -1);
    const auto& samplesPerTriangle = parser.add<int>("samples-per-triangle", "Force the scene to be sampled with the given number of samples per triangle", '\0', arrrgh::Optional, 30);


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

    if(forceGPU.value() != -1) {
        cudaSetDevice(forceGPU.value());
    }

    std::cout << "Listing object directory..";
    std::vector<std::string> objectFileList = listDir(objectDir.value());
    std::cout << " (found " << objectFileList.size() << " files)" << std::endl;

    std::cout << "Listing result directory..";
    std::vector<std::string> resultFileList = listDir(resultDir.value());
    std::cout << " (found " << resultFileList.size() << " files)" << std::endl;
    std::cout << std::endl;

    std::sort(resultFileList.begin(), resultFileList.end());

    std::vector<std::string> parts;
    unsigned int firstIndex = (computeSingleIndex.value() != -1) ? computeSingleIndex.value() : startIndex.value();
    unsigned int lastIndex = (computeSingleIndex.value() != -1) ? computeSingleIndex.value() + 1 : resultFileList.size();

    for (unsigned int resultFileIndex = firstIndex; resultFileIndex < lastIndex; resultFileIndex++) {
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
        if(resultFileContents["overrideObjectCount"] != -1) {
            sampleObjectCount = resultFileContents["overrideObjectCount"];
        }
        float boxSize = float(resultFileContents["boxSize"]);

        std::vector<ShapeDescriptor::cpu::Mesh> objects;
        objects.resize(sampleObjectCount);

        // Normally objects are chosen randomly, and put into a random order. Here, we instead just
        // read the objects in the shuffled order from the result dump file
        for(unsigned int object = 0; object < sampleObjectCount; object++) {
            std::string objectFile = resultFileContents["inputFiles"].at(object);
            stringSplit(&parts, objectFile, '/');
            std::cout << "\t" << "Loading object " << (object + 1) << "/" << sampleObjectCount << ": " << parts.at(parts.size() - 1) << std::endl;
            objects.at(object) = ShapeDescriptor::utilities::loadOBJ(objectDir.value() + "/" + parts.at(parts.size() - 1));
        }
        std::cout << std::endl;

        // 5 Scale all models to fit in a 1x1x1 sphere
        std::cout << "\tScaling meshes.." << std::endl;
        std::vector<ShapeDescriptor::cpu::Mesh> scaledMeshes(sampleObjectCount);
        for (unsigned int i = 0; i < sampleObjectCount; i++) {
            scaledMeshes.at(i) = ShapeDescriptor::utilities::fitMeshInsideSphereOfRadius(objects.at(i), 1);
            ShapeDescriptor::free::mesh(objects.at(i));
        }

        // We need to reduce the scene to the number of objects we want to test with
        int numberOfObjectsInExperiment = *std::max_element(resultFileContents["sampleObjectCounts"].begin(), resultFileContents["sampleObjectCounts"].end());
        // Avoid a memory leak
        for(int i = numberOfObjectsInExperiment; i < scaledMeshes.size(); i++) {
            ShapeDescriptor::free::mesh(scaledMeshes.at(i));
        }
        std::cout << "Constructing a scene with " << numberOfObjectsInExperiment << " objects.." << std::endl;
        scaledMeshes.resize(numberOfObjectsInExperiment);
        sampleObjectCount = numberOfObjectsInExperiment;

        // 6 Copy meshes to GPU
        std::cout << "\tCopying meshes to device.." << std::endl;
        std::vector<ShapeDescriptor::gpu::Mesh> scaledMeshesOnGPU(sampleObjectCount);
        for (unsigned int i = 0; i < sampleObjectCount; i++) {
            scaledMeshesOnGPU.at(i) = ShapeDescriptor::copy::hostMeshToDevice(scaledMeshes.at(i));
        }


        // 10 Combine meshes into one larger scene
        ShapeDescriptor::gpu::Mesh boxScene = combineMeshesOnGPU(scaledMeshesOnGPU);

        // 11 Compute unique vertex mapping
        std::vector<size_t> uniqueVertexCounts;
        size_t totalUniqueVertexCount;
        ShapeDescriptor::gpu::array<signed long long> device_indexMapping = ShapeDescriptor::utilities::computeUniqueIndexMapping(boxScene, scaledMeshesOnGPU, &uniqueVertexCounts, totalUniqueVertexCount);

        // 12 Randomly transform objects
        std::cout << "\tRandomly transforming input objects.." << std::endl;
        std::vector<Transformation> transformations;

        for(int i = 0; i < scaledMeshesOnGPU.size(); i++) {
            Transformation trans{};
            // THESE ROTATION AXES ARE WRONG ON PURPOSE
            // Output dump file contains incorrect rotation order: rotations.at(i) = glm::vec3(yaw, pitch, roll);
            trans.rotation.x = resultFileContents["rotations"][i][1];
            trans.rotation.y = resultFileContents["rotations"][i][0];
            trans.rotation.z = resultFileContents["rotations"][i][2];

            trans.position.x = resultFileContents["translations"][i][0];
            trans.position.y = resultFileContents["translations"][i][1];
            trans.position.z = resultFileContents["translations"][i][2];

            transformations.push_back(trans);
        }

        randomlyTransformMeshes(boxScene, scaledMeshesOnGPU, transformations);


        // 13 Compute corresponding transformed vertex buffer
        //    A mapping is used here because the previously applied transformation can cause non-unique vertices to become
        //    equivalent. It is vital we can rely on a 1:1 mapping existing between vertices.
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_uniqueSpinOrigins = ShapeDescriptor::utilities::applyUniqueMapping(boxScene, device_indexMapping, totalUniqueVertexCount);
        checkCudaErrors(cudaFree(device_indexMapping.content));

        // Should be as large as possible, but can be selected arbitrarily
        size_t sampleCount = std::max(samplesPerTriangle.value() * (boxScene.vertexCount / 3), (size_t) 1000000);

        std::cout << "\tSampling scene.. (" << sampleCount << " samples)" << std::endl;
        ShapeDescriptor::internal::MeshSamplingBuffers sampleBuffers;

        // 1 Seeding the random number generator
        std::random_device rd;
        std::minstd_rand0 generator{rd()};

        ShapeDescriptor::gpu::PointCloud sampledScene = ShapeDescriptor::utilities::sampleMesh(boxScene, sampleCount, generator(), &sampleBuffers);

        std::cout << "\tComputing reference object sample count.." << std::endl;
        size_t referenceObjectSampleCount = computeReferenceSampleCount(scaledMeshesOnGPU.at(0), sampleCount, sampleBuffers.cumulativeAreaArray);
        std::cout << "\t\tReference object has " << referenceObjectSampleCount << " samples." << std::endl;

        std::cout << "\tComputing clutter values.." << std::endl;
        float spinImageWidth = resultFileContents["spinImageWidth"];

        ShapeDescriptor::cpu::array<float> clutterValues = computeClutter(device_uniqueSpinOrigins, sampledScene, spinImageWidth, referenceObjectSampleCount, resultFileContents["uniqueVertexCounts"][0]);

        json outJson;

        outJson["version"] = "clutter_v1";
        outJson["clutterValues"] = {};
        outJson["sourceFile"] = resultFile;
        outJson["seed"] = resultFileContents["seed"];
        outJson["sampleObjectCount"] = resultFileContents["sampleObjectCount"];
        outJson["sampleCount"] = sampleCount;

        for(size_t item = 0; item < clutterValues.length; item++) {
            outJson["clutterValues"].push_back(clutterValues.content[item]);
        }

        std::string outFilePath = outDir.value() + "/" + getCurrentDateTimeString() + "_" + std::to_string((int)resultFileContents["seed"]) + ".json";
        std::cout << "Writing output file to " << outFilePath << std::endl;
        std::ofstream outFile(outFilePath);
        outFile << outJson.dump(4) << std::endl;
        outFile.close();

        std::cout << "Cleaning up.." << std::endl;
        sampledScene.free();
        cudaFree(device_uniqueSpinOrigins.content);
        ShapeDescriptor::gpu::freeMesh(boxScene);

        for(ShapeDescriptor::gpu::Mesh deviceMesh : scaledMeshesOnGPU) {
            ShapeDescriptor::gpu::freeMesh(deviceMesh);
        }

        std::cout << "Done." << std::endl;
    }

}