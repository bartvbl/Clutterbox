#include "3DSCMethod.h"
#include <spinImage/gpu/3dShapeContextGenerator.cuh>
#include <spinImage/gpu/3dShapeContextSearcher.cuh>

SpinImage::gpu::array<char> SCMethod::generateDescriptors(
        SpinImage::gpu::Mesh device_sceneAsMesh,
        SpinImage::gpu::PointCloud device_sceneAsPointCloud,
        SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::SCExecutionTimes scExecutionTimes{};

    SpinImage::gpu::array<SpinImage::gpu::ShapeContextDescriptor> descriptors = SpinImage::gpu::generate3DSCDescriptors(
            device_sceneAsPointCloud,
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

SpinImage::cpu::array<unsigned int> SCMethod::computeSearchResultRanks(
        SpinImage::gpu::array<char> device_needleDescriptors,
        SpinImage::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::SCSearchExecutionTimes times{};

    SpinImage::cpu::array<unsigned int> searchResultIndices = SpinImage::gpu::compute3DSCSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<SpinImage::gpu::ShapeContextDescriptor*>(device_needleDescriptors.content)},
             parameters.needleDescriptorScenePointCloudPointCount,
            {device_haystackDescriptors.length,
             reinterpret_cast<SpinImage::gpu::ShapeContextDescriptor*>(device_haystackDescriptors.content)},
             parameters.haystackDescriptorScenePointCloudPointCount,
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

const std::string SCMethod::getMethodCommandLineParameterName() {
    return "3dsc";
}

const std::string SCMethod::getMethodDumpFileName() {
    return "3DSC";
}
