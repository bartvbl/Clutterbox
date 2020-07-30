#include "RICIMethod.h"
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>

SpinImage::gpu::array<char> RICIMethod::generateDescriptors(
        SpinImage::gpu::Mesh device_sceneAsMesh,
        SpinImage::gpu::PointCloud device_sceneAsPointCloud,
        SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::RICIExecutionTimes riciExecutionTimes{};

    SpinImage::gpu::array<SpinImage::gpu::RICIDescriptor> descriptors = SpinImage::gpu::generateRadialIntersectionCountImages(
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

SpinImage::cpu::array<unsigned int> RICIMethod::computeSearchResultRanks(
        SpinImage::gpu::array<char> device_needleDescriptors,
        SpinImage::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::RICISearchExecutionTimes times{};

    SpinImage::cpu::array<unsigned int> searchResultIndices = SpinImage::gpu::computeRadialIntersectionCountImageSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<SpinImage::gpu::RICIDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<SpinImage::gpu::RICIDescriptor*>(device_haystackDescriptors.content)},
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
