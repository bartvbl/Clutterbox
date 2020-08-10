#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include "SIMethod.h"

ShapeDescriptor::gpu::array<char> SIMethod::generateDescriptors(
        ShapeDescriptor::gpu::Mesh device_sceneAsMesh,
        ShapeDescriptor::gpu::PointCloud device_sceneAsPointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::SIExecutionTimes siExecutionTimes{};

    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors = ShapeDescriptor::gpu::generateSpinImages(
            device_sceneAsPointCloud,
            device_descriptorOrigins,
            parameters.supportRadius,
            supportAngle,
            &siExecutionTimes);

    executionTimes->append("total", siExecutionTimes.totalExecutionTimeSeconds);
    executionTimes->append("initialisation", siExecutionTimes.initialisationTimeSeconds);
    executionTimes->append("generation", siExecutionTimes.generationTimeSeconds);

    return {descriptors.length, reinterpret_cast<char*>(descriptors.content)};
}

ShapeDescriptor::cpu::array<unsigned int> SIMethod::computeSearchResultRanks(
        ShapeDescriptor::gpu::array<char> device_needleDescriptors,
        ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::SISearchExecutionTimes times{};

    ShapeDescriptor::cpu::array<unsigned int> searchResultIndices = ShapeDescriptor::gpu::computeSpinImageSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<ShapeDescriptor::SpinImageDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<ShapeDescriptor::SpinImageDescriptor*>(device_haystackDescriptors.content)},
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("averaging", times.averagingExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

void SIMethod::dumpMetadata(json jsonOutput) {
    jsonOutput["spinImageSupportAngle"] = supportAngle;
    jsonOutput["spinImageWidthPixels"] = spinImageWidthPixels;
}

const std::string SIMethod::getMethodCommandLineParameterName() {
    return "si";
}

const std::string SIMethod::getMethodDumpFileName() {
    return "SI";
}
