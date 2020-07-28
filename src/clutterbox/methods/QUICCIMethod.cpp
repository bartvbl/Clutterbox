#include "QUICCIMethod.h"
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>
#include <spinImage/gpu/quickIntersectionCountImageSearcher.cuh>

SpinImage::array<char> QUICCIMethod::generateDescriptors(
        SpinImage::gpu::Mesh device_sceneAsMesh,
        SpinImage::gpu::PointCloud device_sceneAsPointCloud,
        SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::QUICCIExecutionTimes quicciExecutionTimes{};

    SpinImage::array<SpinImage::gpu::QUICCIDescriptor> descriptors = SpinImage::gpu::generateQUICCImages(
            device_sceneAsMesh,
            device_descriptorOrigins,
            parameters.supportRadius,
            &quicciExecutionTimes);

    executionTimes->append("total", quicciExecutionTimes.totalExecutionTimeSeconds);
    executionTimes->append("meshScale", quicciExecutionTimes.meshScaleTimeSeconds);
    executionTimes->append("redistribution", quicciExecutionTimes.redistributionTimeSeconds);
    executionTimes->append("generation", quicciExecutionTimes.generationTimeSeconds);

    return {descriptors.length, reinterpret_cast<char*>(descriptors.content)};
}

SpinImage::array<unsigned int> QUICCIMethod::computeSearchResultRanks(
        SpinImage::array<char> device_needleDescriptors,
        SpinImage::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::QUICCISearchExecutionTimes times{};

    SpinImage::array<unsigned int> searchResultIndices = SpinImage::gpu::computeQUICCImageSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<SpinImage::gpu::QUICCIDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<SpinImage::gpu::QUICCIDescriptor*>(device_haystackDescriptors.content)},
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

const std::string QUICCIMethod::getMethodCommandLineParameterName() {
    return "quicci";
}

const std::string QUICCIMethod::getMethodDumpFileName() {
    return "QUICCI";
}
