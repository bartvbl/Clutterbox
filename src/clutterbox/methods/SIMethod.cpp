#include <spinImage/gpu/spinImageGenerator.cuh>
#include <spinImage/gpu/spinImageSearcher.cuh>
#include "SIMethod.h"

SpinImage::gpu::array<char> SIMethod::generateDescriptors(
        SpinImage::gpu::Mesh device_sceneAsMesh,
        SpinImage::gpu::PointCloud device_sceneAsPointCloud,
        SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::SIExecutionTimes siExecutionTimes{};

    SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> descriptors = SpinImage::gpu::generateSpinImages(
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

SpinImage::cpu::array<unsigned int> SIMethod::computeSearchResultRanks(
        SpinImage::gpu::array<char> device_needleDescriptors,
        SpinImage::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::SISearchExecutionTimes times{};

    SpinImage::cpu::array<unsigned int> searchResultIndices = SpinImage::gpu::computeSpinImageSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<SpinImage::gpu::SpinImageDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<SpinImage::gpu::SpinImageDescriptor*>(device_haystackDescriptors.content)},
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("averaging", times.averagingExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

const std::string SIMethod::getMethodCommandLineParameterName() {
    return "si";
}

const std::string SIMethod::getMethodDumpFileName() {
    return "SI";
}
