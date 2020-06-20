#include "quicciDistanceFunctionBenchmark.h"
#include "clutterSphereMeshAugmenter.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/utilities/modelScaler.h>
#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/utilities/duplicateRemoval.cuh>
#include <utilities/randomFileSelector.h>
#include <cuda_runtime_api.h>
#include <nvidia/helper_cuda.h>
#include <experiments/clutterBox/clutterBoxUtilities.h>
#include <spinImage/utilities/dumpers/meshDumper.h>

void runQuicciDistanceFunctionBenchmark(
        std::string sourceDirectory,
        std::string outputDirectory,
        size_t seed,
        std::vector<int> sphereCountList,
        int sceneSphereCount,
        float clutterSphereRadius) {
    // 1 Seeding the random number generator
    std::random_device rd("/dev/urandom");
    size_t randomSeed = seed != 0 ? seed : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::minstd_rand0 generator{randomSeed};

    std::cout << std::endl << "Running experiment initialisation sequence.." << std::endl;

    // 2 Search SHREC directory for files
    // 3 Make a sample set of n sample objects
    std::vector<std::string> filePaths = generateRandomFileList(sourceDirectory, 1, generator);

    // 4 Load the models in the sample set
    std::cout << "\tLoading sample model.." << std::endl;
    SpinImage::cpu::Mesh sampleMesh = SpinImage::utilities::loadOBJ(filePaths.at(0), true);

    // 5 Scale all models to fit in a 1x1x1 sphere
    std::cout << "\tScaling meshes.." << std::endl;
    SpinImage::cpu::Mesh scaledMesh = SpinImage::utilities::fitMeshInsideSphereOfRadius(sampleMesh, 1);
    SpinImage::cpu::freeMesh(sampleMesh);

    // 6 Add clutter spheres to the mesh
    std::cout << "\tAugmenting mesh with spheres.." << std::endl;
    SpinImage::cpu::Mesh augmentedMesh = applyClutterSpheres(scaledMesh, sceneSphereCount, clutterSphereRadius, generator());
    SpinImage::cpu::freeMesh(scaledMesh);

    std::cout << "DUMPING MESH "<< std::endl;
    SpinImage::dump::mesh(augmentedMesh, "sphereClutter.obj");

    // 6 Copy meshes to GPU
    std::cout << "\tCopying meshes to device.." << std::endl;
    SpinImage::gpu::Mesh scaledMeshOnGPU = SpinImage::copy::hostMeshToDevice(augmentedMesh);

    // 8 Remove duplicate vertices
    std::cout << "\tRemoving duplicate vertices.." << std::endl;
    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> spinOrigins_reference = SpinImage::utilities::computeUniqueVertices(scaledMeshOnGPU);
    size_t referenceImageCount = spinOrigins_reference.length;
    std::cout << "\t\tReduced " << scaledMeshOnGPU.vertexCount << " vertices to " << referenceImageCount << "." << std::endl;

    std::cout << "\tGenerating QUICCI images.." << std::endl;
/*
    SpinImage::debug::QUICCIRunInfo quicciReferenceRunInfo;
    device_referenceQuiccImages = SpinImage::gpu::generateQUICCImages(
            scaledMeshesOnGPU.at(0),
            spinOrigins_reference,
            supportRadius,
            &quicciReferenceRunInfo);

    QUICCIRuns.push_back(quicciReferenceRunInfo);
    std::cout << "\t\tExecution time: " << quicciReferenceRunInfo.generationTimeSeconds << std::endl;

    checkCudaErrors(cudaFree(spinOrigins_reference.content));

    // 10 Combine meshes into one larger scene
    SpinImage::gpu::Mesh boxScene = combineMeshesOnGPU(scaledMeshOnGPU);

    // 11 Compute unique vertex mapping
    std::vector<size_t> uniqueVertexCounts;
    size_t totalUniqueVertexCount;
    SpinImage::array<signed long long> device_indexMapping = SpinImage::utilities::computeUniqueIndexMapping(boxScene, scaledMeshesOnGPU, &uniqueVertexCounts, totalUniqueVertexCount);

    // 12 Randomly transform objects
    std::cout << "\tRandomly transforming input objects.." << std::endl;
    randomlyTransformMeshes(boxScene, boxSize, scaledMeshesOnGPU, generator);

    size_t vertexCount = 0;
    size_t referenceMeshImageCount = spinOrigins_reference.length;

    // 13 Compute corresponding transformed vertex buffer
    //    A mapping is used here because the previously applied transformation can cause non-unique vertices to become
    //    equivalent. It is vital we can rely on a 1:1 mapping existing between vertices.
    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_uniqueSpinOrigins = SpinImage::utilities::applyUniqueMapping(boxScene, device_indexMapping, totalUniqueVertexCount);
    checkCudaErrors(cudaFree(device_indexMapping.content));
    size_t imageCount = 0;

    // 14 Ensure enough memory is available to complete the experiment.
    if(riciDescriptorActive || quicciDescriptorActive || siDescriptorActive || shapeContextDescriptorActive) {
        std::cout << "\tTesting for sufficient memory capacity on GPU.. ";
        int* device_largestNecessaryImageBuffer;
        size_t largestImageBufferSize = totalUniqueVertexCount * spinImageWidthPixels * spinImageWidthPixels * sizeof(int);
        if(quicciDescriptorActive && !riciDescriptorActive && !siDescriptorActive && !shapeContextDescriptorActive) {
            // QUICCI only needs one bit per pixel, thus way less memory
            largestImageBufferSize /= 8;
        }
        checkCudaErrors(cudaMalloc((void**) &device_largestNecessaryImageBuffer, largestImageBufferSize));
        checkCudaErrors(cudaFree(device_largestNecessaryImageBuffer));
        std::cout << "Success." << std::endl;
    }

    std::vector<SpinImage::array<unsigned int>> rawRICISearchResults;
    std::vector<SpinImage::array<unsigned int>> rawQUICCISearchResults;
    std::vector<SpinImage::array<unsigned int>> rawSISearchResults;
    std::vector<SpinImage::array<unsigned int>> rawFPFHSearchResults;
    std::vector<SpinImage::array<unsigned int>> raw3DSCSearchResults;
    std::vector<size_t> spinImageSampleCounts;

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


        // Generating radial intersection count images
        if(riciDescriptorActive) {
            std::cout << "\tGenerating RICI images.. (" << imageCount << " images)" << std::endl;
            SpinImage::debug::RICIRunInfo riciSampleRunInfo;
            SpinImage::array<radialIntersectionCountImagePixelType> device_sampleRICIImages = SpinImage::gpu::generateRadialIntersectionCountImages(
                    boxScene,
                    device_uniqueSpinOrigins,
                    supportRadius,
                    &riciSampleRunInfo);
            RICIRuns.push_back(riciSampleRunInfo);
            std::cout << "\t\tTimings: (total " << riciSampleRunInfo.totalExecutionTimeSeconds
                      << ", scaling " << riciSampleRunInfo.meshScaleTimeSeconds
                      << ", redistribution " << riciSampleRunInfo.redistributionTimeSeconds
                      << ", generation " << riciSampleRunInfo.generationTimeSeconds << ")" << std::endl;

            std::cout << "\tSearching in radial intersection count images.." << std::endl;
            SpinImage::debug::RICISearchRunInfo riciSearchRun;
            SpinImage::array<unsigned int> RICIsearchResults = SpinImage::gpu::computeRadialIntersectionCountImageSearchResultRanks(
                    device_referenceRICIImages,
                    referenceMeshImageCount,
                    device_sampleRICIImages,
                    imageCount,
                    &riciSearchRun);
            RICISearchRuns.push_back(riciSearchRun);
            rawRICISearchResults.push_back(RICIsearchResults);
            std::cout << "\t\tTimings: (total " << riciSearchRun.totalExecutionTimeSeconds
                      << ", searching " << riciSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
            Histogram RICIHistogram = computeSearchResultHistogram(referenceMeshImageCount, RICIsearchResults);

            if(enableMatchVisualisation && std::find(matchVisualisationDescriptorList.begin(), matchVisualisationDescriptorList.end(), "rici") != matchVisualisationDescriptorList.end()) {
                std::cout << "\tDumping OBJ visualisation of search results.." << std::endl;
                std::experimental::filesystem::path outFilePath = matchVisualisationOutputDir;
                outFilePath = outFilePath / (std::to_string(randomSeed) + "_rici_" + std::to_string(objectCount + 1) + ".obj");
                dumpSearchResultVisualisationMesh(RICIsearchResults, scaledMeshesOnGPU.at(0), outFilePath);
            }

            if(!dumpRawSearchResults) {
                delete[] RICIsearchResults.content;
            }

            // Storing results
            RICIHistograms.push_back(RICIHistogram);

            // Finally, delete the RICI descriptor images
            cudaFree(device_sampleRICIImages.content);
        }

        if(quicciDescriptorActive) {
            std::cout << "\tGenerating QUICCI images.. (" << imageCount << " images)" << std::endl;
            SpinImage::debug::QUICCIRunInfo quicciSampleRunInfo;
            SpinImage::gpu::QUICCIImages device_sampleQUICCImages = SpinImage::gpu::generateQUICCImages(
                    boxScene,
                    device_uniqueSpinOrigins,
                    supportRadius,
                    &quicciSampleRunInfo);
            QUICCIRuns.push_back(quicciSampleRunInfo);
            std::cout << "\t\tTimings: (total " << quicciSampleRunInfo.totalExecutionTimeSeconds
                      << ", generation " << quicciSampleRunInfo.generationTimeSeconds << ")" << std::endl;

            std::cout << "\tSearching in QUICC images.." << std::endl;
            SpinImage::debug::QUICCISearchRunInfo quicciSearchRun;
            SpinImage::array<unsigned int> QUICCIsearchResults = SpinImage::gpu::computeQUICCImageSearchResultRanks(
                    device_referenceQuiccImages,
                    referenceMeshImageCount,
                    device_sampleQUICCImages,
                    imageCount,
                    &quicciSearchRun);
            QUICCISearchRuns.push_back(quicciSearchRun);
            rawQUICCISearchResults.push_back(QUICCIsearchResults);
            std::cout << "\t\tTimings: (total " << quicciSearchRun.totalExecutionTimeSeconds
                      << ", searching " << quicciSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
            Histogram QUICCIHistogram = computeSearchResultHistogram(referenceMeshImageCount, QUICCIsearchResults);
            cudaFree(device_sampleQUICCImages.images);

            if(enableMatchVisualisation && std::find(matchVisualisationDescriptorList.begin(), matchVisualisationDescriptorList.end(), "quicci") != matchVisualisationDescriptorList.end()) {
                std::cout << "\tDumping OBJ visualisation of search results.." << std::endl;
                std::experimental::filesystem::path outFilePath = matchVisualisationOutputDir;
                outFilePath = outFilePath / (std::to_string(randomSeed) + "_quicci_" + std::to_string(objectCount + 1) + ".obj");
                dumpSearchResultVisualisationMesh(QUICCIsearchResults, scaledMeshesOnGPU.at(0), outFilePath);
            }

            if(!dumpRawSearchResults) {
                delete[] QUICCIsearchResults.content;
            }

            // Storing results
            QUICCIHistograms.push_back(QUICCIHistogram);
        }

        // Computing common settings for SI, FPFH, and 3DSC
        size_t meshSamplingSeed = generator();
        size_t currentReferenceObjectSampleCount = 0;
        if(siDescriptorActive || shapeContextDescriptorActive || fastPointFeatureHistogramActive) {
            spinImageSampleCount = computeSpinImageSampleCount(imageCount);
            spinImageSampleCounts.push_back(spinImageSampleCount);

            // wasteful solution, but I don't want to do ugly hacks that destroy the function API's
            // Computes number of samples used for the reference object
            SpinImage::internal::MeshSamplingBuffers sampleBuffers;
            SpinImage::gpu::PointCloud device_pointCloud = SpinImage::utilities::sampleMesh(boxScene, spinImageSampleCount, meshSamplingSeed, &sampleBuffers);
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
            currentReferenceObjectSampleCount = size_t(double(areaFraction) * double(spinImageSampleCount));
            std::cout << "\t\tReference object sample count: " << currentReferenceObjectSampleCount << std::endl;
        }


        // Generating spin images
        if(siDescriptorActive) {
            std::cout << "\tGenerating spin images.. (" << imageCount << " images, " << spinImageSampleCount << " samples)" << std::endl;
            SpinImage::debug::SIRunInfo siSampleRunInfo;
            SpinImage::array<spinImagePixelType> device_sampleSpinImages = SpinImage::gpu::generateSpinImages(
                    boxScene,
                    device_uniqueSpinOrigins,
                    supportRadius,
                    spinImageSampleCount,
                    spinImageSupportAngleDegrees,
                    meshSamplingSeed,
                    &siSampleRunInfo);
            SIRuns.push_back(siSampleRunInfo);
            std::cout << "\t\tTimings: (total " << siSampleRunInfo.totalExecutionTimeSeconds
                      << ", initialisation " << siSampleRunInfo.initialisationTimeSeconds
                      << ", sampling " << siSampleRunInfo.meshSamplingTimeSeconds
                      << ", generation " << siSampleRunInfo.generationTimeSeconds << ")" << std::endl;

            std::cout << "\tSearching in spin images.." << std::endl;
            SpinImage::debug::SISearchRunInfo siSearchRun;
            SpinImage::array<unsigned int> SpinImageSearchResults = SpinImage::gpu::computeSpinImageSearchResultRanks(
                    device_referenceSpinImages,
                    referenceMeshImageCount,
                    device_sampleSpinImages,
                    imageCount,
                    &siSearchRun);
            SISearchRuns.push_back(siSearchRun);
            rawSISearchResults.push_back(SpinImageSearchResults);
            std::cout << "\t\tTimings: (total " << siSearchRun.totalExecutionTimeSeconds
                      << ", averaging " << siSearchRun.averagingExecutionTimeSeconds
                      << ", searching " << siSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
            Histogram SIHistogram = computeSearchResultHistogram(referenceMeshImageCount, SpinImageSearchResults);
            cudaFree(device_sampleSpinImages.content);

            if(enableMatchVisualisation && std::find(matchVisualisationDescriptorList.begin(), matchVisualisationDescriptorList.end(), "si") != matchVisualisationDescriptorList.end()) {
                std::cout << "\tDumping OBJ visualisation of search results.." << std::endl;
                std::experimental::filesystem::path outFilePath = matchVisualisationOutputDir;
                outFilePath = outFilePath / (std::to_string(randomSeed) + "_si_" + std::to_string(objectCount + 1) + ".obj");
                dumpSearchResultVisualisationMesh(SpinImageSearchResults, scaledMeshesOnGPU.at(0), outFilePath);
            }

            if(!dumpRawSearchResults) {
                delete[] SpinImageSearchResults.content;
            }

            // Storing results
            spinImageHistograms.push_back(SIHistogram);
        }


        // Generating 3D Shape Context descriptors
        if(shapeContextDescriptorActive) {
            std::cout << "\tGenerating 3D shape context descriptors.. (" << imageCount << " images, " << spinImageSampleCount << " samples)" << std::endl;
            SpinImage::debug::SCRunInfo scSampleRunInfo;
            SpinImage::array<shapeContextBinType> device_sample3DSCDescriptors = SpinImage::gpu::generate3DSCDescriptors(
                    boxScene,
                    device_uniqueSpinOrigins,
                    pointDensityRadius3dsc,
                    minSupportRadius3dsc,
                    supportRadius,
                    spinImageSampleCount,
                    meshSamplingSeed,
                    &scSampleRunInfo);
            ShapeContextRuns.push_back(scSampleRunInfo);
            std::cout << "\t\tTimings: (total " << scSampleRunInfo.totalExecutionTimeSeconds
                      << ", initialisation " << scSampleRunInfo.initialisationTimeSeconds
                      << ", sampling " << scSampleRunInfo.meshSamplingTimeSeconds
                      << ", point counting " << scSampleRunInfo.pointCountingTimeSeconds
                      << ", generation " << scSampleRunInfo.generationTimeSeconds << ")" << std::endl;

            std::cout << "\tSearching in 3D Shape Context descriptors.." << std::endl;
            SpinImage::debug::SCSearchRunInfo scSearchRun;
            SpinImage::array<unsigned int> ShapeContextSearchResults = SpinImage::gpu::compute3DSCSearchResultRanks(
                    device_referenceShapeContextDescriptors,
                    referenceMeshImageCount,
                    referenceSampleCount,
                    device_sample3DSCDescriptors,
                    imageCount,
                    currentReferenceObjectSampleCount,
                    &scSearchRun);
            ShapeContextSearchRuns.push_back(scSearchRun);
            raw3DSCSearchResults.push_back(ShapeContextSearchResults);
            std::cout << "\t\tTimings: (total " << scSearchRun.totalExecutionTimeSeconds
                      << ", searching " << scSearchRun.searchExecutionTimeSeconds << ")" << std::endl;
            Histogram SCHistogram = computeSearchResultHistogram(referenceMeshImageCount, ShapeContextSearchResults);
            cudaFree(device_sample3DSCDescriptors.content);

            if(enableMatchVisualisation && std::find(matchVisualisationDescriptorList.begin(), matchVisualisationDescriptorList.end(), "3dsc") != matchVisualisationDescriptorList.end()) {
                std::cout << "\tDumping OBJ visualisation of search results.." << std::endl;
                std::experimental::filesystem::path outFilePath = matchVisualisationOutputDir;
                outFilePath = outFilePath / (std::to_string(randomSeed) + "_3dsc_" + std::to_string(objectCount + 1) + ".obj");
                dumpSearchResultVisualisationMesh(ShapeContextSearchResults, scaledMeshesOnGPU.at(0), outFilePath);
            }

            if(!dumpRawSearchResults) {
                delete[] ShapeContextSearchResults.content;
            }

            // Storing results
            shapeContextHistograms.push_back(SCHistogram);
        }


        // Generating Fast Point Feature Histograms
        if(fastPointFeatureHistogramActive) {
            std::cout << "\tGenerating Fast Point Feature Histograms.. (" << imageCount << " images, " << spinImageSampleCount << " samples)" << std::endl;
            SpinImage::debug::FPFHRunInfo fpfhSampleRunInfo;
            SpinImage::gpu::FPFHHistograms device_sampleFPFHHistograms = SpinImage::gpu::generateFPFHHistograms(
                    boxScene,
                    device_uniqueSpinOrigins,
                    supportRadius,
                    fpfhBinCount,
                    spinImageSampleCount,
                    meshSamplingSeed,
                    &fpfhSampleRunInfo);

            FPFHRuns.push_back(fpfhSampleRunInfo);
            std::cout << "\t\tTimings: (total " << fpfhSampleRunInfo.totalExecutionTimeSeconds << ")" << std::endl;

            std::cout << "\tSearching in FPFH descriptors.." << std::endl;
            SpinImage::debug::FPFHSearchRunInfo fpfhSearchRun;
            SpinImage::array<unsigned int> FPFHSearchResults = SpinImage::gpu::computeFPFHSearchResultRanks(
                    device_referenceFPFHHistograms,
                    referenceMeshImageCount,
                    device_sampleFPFHHistograms,
                    imageCount,
                    &fpfhSearchRun);
            FPFHSearchRuns.push_back(fpfhSearchRun);
            rawFPFHSearchResults.push_back(FPFHSearchResults);
            std::cout << "\t\tTimings: (total " << fpfhSearchRun.totalExecutionTimeSeconds << ")" << std::endl;
            Histogram FPFHHistogram = computeSearchResultHistogram(referenceMeshImageCount, FPFHSearchResults);
            cudaFree(device_sampleFPFHHistograms.histograms);

            if(enableMatchVisualisation && std::find(matchVisualisationDescriptorList.begin(), matchVisualisationDescriptorList.end(), "fpfh") != matchVisualisationDescriptorList.end()) {
                std::cout << "\tDumping OBJ visualisation of search results.." << std::endl;
                std::experimental::filesystem::path outFilePath = matchVisualisationOutputDir;
                outFilePath = outFilePath / (std::to_string(randomSeed) + "_fpfh_" + std::to_string(objectCount + 1) + ".obj");
                dumpSearchResultVisualisationMesh(FPFHSearchResults, scaledMeshesOnGPU.at(0), outFilePath);
            }

            if(!dumpRawSearchResults) {
                delete[] FPFHSearchResults.content;
            }

            // Storing results
            FPFHHistograms.push_back(FPFHHistogram);
        }


        // Dumping OBJ file of current scene, if enabled
        if(dumpSceneOBJFiles) {
            SpinImage::cpu::Mesh hostMesh = SpinImage::copy::deviceMeshToHost(boxScene);

            std::experimental::filesystem::path outFilePath = sceneOBJFileDumpDir;
            outFilePath = outFilePath / (std::to_string(randomSeed) + "_" + std::to_string(objectCount + 1) + ".obj");

            std::cout << "\tDumping OBJ file of scene to " << outFilePath << std::endl;

            SpinImage::dump::mesh(hostMesh, outFilePath, 0, scaledMeshesOnGPU.at(0).vertexCount);

            SpinImage::cpu::freeMesh(hostMesh);
        }
    }

    SpinImage::gpu::freeMesh(boxScene);
    cudaFree(device_referenceRICIImages.content);
    cudaFree(device_referenceSpinImages.content);
    cudaFree(device_referenceQuiccImages.images);
    cudaFree(device_uniqueSpinOrigins.content);

    std::string timestring = getCurrentDateTimeString();

    dumpResultsFile(
            outputDirectory + timestring + "_" + std::to_string(randomSeed) + ".json",
            descriptorList,
            randomSeed,
            RICIHistograms,
            QUICCIHistograms,
            spinImageHistograms,
            shapeContextHistograms,
            FPFHHistograms,
            objectDirectory,
            objectCountList,
            overrideObjectCount,
            boxSize,
            supportRadius,
            minSupportRadius3dsc,
            pointDensityRadius3dsc,
            fpfhBinCount,
            generator(),
            RICIRuns,
            QUICCIRuns,
            SIRuns,
            ShapeContextRuns,
            FPFHRuns,
            RICISearchRuns,
            QUICCISearchRuns,
            SISearchRuns,
            ShapeContextSearchRuns,
            FPFHSearchRuns,
            spinImageSupportAngleDegrees,
            uniqueVertexCounts,
            spinImageSampleCounts,
            gpuMetaData);

    if(dumpRawSearchResults) {
        dumpRawSearchResultFile(
                outputDirectory + "raw/" + timestring + "_" + std::to_string(randomSeed) + ".json",
                descriptorList,
                objectCountList,
                rawRICISearchResults,
                rawQUICCISearchResults,
                rawSISearchResults,
                raw3DSCSearchResults,
                rawFPFHSearchResults,
                randomSeed);

        // Cleanup
        // If one of the descriptors is not enabled, this will iterate over an empty vector.
        for(auto results : rawRICISearchResults) {
            delete[] results.content;
        }
        for(auto results : rawQUICCISearchResults) {
            delete[] results.content;
        }
        for(auto results : rawSISearchResults) {
            delete[] results.content;
        }
        for(auto results : rawFPFHSearchResults) {
            delete[] results.content;
        }
        for(auto results : raw3DSCSearchResults) {
            delete[] results.content;
        }
    }

    for(SpinImage::gpu::Mesh deviceMesh : scaledMeshesOnGPU) {
        SpinImage::gpu::freeMesh(deviceMesh);
    }

    std::cout << std::endl << "Complete." << std::endl;*/
}