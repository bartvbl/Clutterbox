#include "3DSCMethod.h"
#include <spinImage/gpu/3dShapeContextGenerator.cuh>
#include <spinImage/gpu/3dShapeContextSearcher.cuh>

SpinImage::array<char> SCMethod::generateDescriptors(
        SpinImage::gpu::Mesh device_sceneMesh,
        SpinImage::gpu::PointCloud device_scenePointCloud,
        SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::SCExecutionTimes scExecutionTimes{};

    SpinImage::array<SpinImage::gpu::ShapeContextDescriptor> descriptors = SpinImage::gpu::generate3DSCDescriptors(
            device_scenePointCloud,
            device_descriptorOrigins,
            pointDensityRadius,
            minSupportRadius,
            parameters.supportRadius,
            &scExecutionTimes);

    executionTimes->append("total", scExecutionTimes.totalExecutionTimeSeconds);
    executionTimes->append("pointCounting", scExecutionTimes.pointCountingTimeSeconds);
    executionTimes->append("initialisation", scExecutionTimes.initialisationTimeSeconds);
    executionTimes->append("generation", scExecutionTimes.generationTimeSeconds);

    return {descriptors.length, reinterpret_cast<char*>(descriptors.content)};
}

SpinImage::array<unsigned int> SCMethod::computeSearchResultRanks(
        SpinImage::array<char> device_needleDescriptors,
        SpinImage::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::SCSearchExecutionTimes times{};

    SpinImage::array<unsigned int> searchResultIndices = SpinImage::gpu::compute3DSCSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<SpinImage::gpu::ShapeContextDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<SpinImage::gpu::ShapeContextDescriptor*>(device_haystackDescriptors.content)},
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

const std::string SCMethod::getMethodCommandLineParameterName() {
    return "rici";
}

const std::string SCMethod::getMethodDumpFileName() {
    return "RICI";
}
