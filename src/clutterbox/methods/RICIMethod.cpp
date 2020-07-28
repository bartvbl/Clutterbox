#include "RICIMethod.h"
#include <spinImage/gpu/radialIntersectionCountImageGenerator.cuh>
#include <spinImage/gpu/radialIntersectionCountImageSearcher.cuh>

SpinImage::array<char> RICIMethod::generateDescriptors(
        SpinImage::gpu::Mesh device_sceneMesh,
        SpinImage::gpu::PointCloud device_scenePointCloud,
        SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        float supportRadius,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::RICIExecutionTimes riciExecutionTimes{};

    SpinImage::array<SpinImage::gpu::RICIDescriptor> descriptors = SpinImage::gpu::generateRadialIntersectionCountImages(
            device_sceneMesh,
            device_descriptorOrigins,
            supportRadius,
            &riciExecutionTimes);

    executionTimes->append("total", riciExecutionTimes.totalExecutionTimeSeconds);
    executionTimes->append("meshScale", riciExecutionTimes.meshScaleTimeSeconds);
    executionTimes->append("redistribution", riciExecutionTimes.redistributionTimeSeconds);
    executionTimes->append("generation", riciExecutionTimes.generationTimeSeconds);

    return {descriptors.length, reinterpret_cast<char*>(descriptors.content)};
}

SpinImage::array<unsigned int> RICIMethod::computeSearchResultRanks(
        SpinImage::array<char> device_needleDescriptors,
        SpinImage::array<char> device_haystackDescriptors,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::RICISearchExecutionTimes times{};

    SpinImage::array<unsigned int> searchResultIndices = SpinImage::gpu::computeRadialIntersectionCountImageSearchResultRanks(
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
