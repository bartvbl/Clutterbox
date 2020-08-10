#include "RICIMethod.h"
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>

ShapeDescriptor::gpu::array<char> RICIMethod::generateDescriptors(
        ShapeDescriptor::gpu::Mesh device_sceneAsMesh,
        ShapeDescriptor::gpu::PointCloud device_sceneAsPointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::RICIExecutionTimes riciExecutionTimes{};

    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors = ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
            device_sceneAsMesh,
            device_descriptorOrigins,
            parameters.supportRadius,
            &riciExecutionTimes);

    executionTimes->append("total", riciExecutionTimes.totalExecutionTimeSeconds);
    executionTimes->append("meshScale", riciExecutionTimes.meshScaleTimeSeconds);
    executionTimes->append("redistribution", riciExecutionTimes.redistributionTimeSeconds);
    executionTimes->append("generation", riciExecutionTimes.generationTimeSeconds);

    return {descriptors.length, reinterpret_cast<char*>(descriptors.content)};
}

ShapeDescriptor::cpu::array<unsigned int> RICIMethod::computeSearchResultRanks(
        ShapeDescriptor::gpu::array<char> device_needleDescriptors,
        ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::RICISearchExecutionTimes times{};

    ShapeDescriptor::cpu::array<unsigned int> searchResultIndices = ShapeDescriptor::gpu::computeRadialIntersectionCountImageSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<ShapeDescriptor::RICIDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<ShapeDescriptor::RICIDescriptor*>(device_haystackDescriptors.content)},
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

const std::string RICIMethod::getMethodCommandLineParameterName() {
    return "rici";
}

const std::string RICIMethod::getMethodDumpFileName() {
    return "RICI";
}
