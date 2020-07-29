#include "FPFHMethod.h"
#include <spinImage/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <spinImage/gpu/fastPointFeatureHistogramSearcher.cuh>

SpinImage::gpu::array<char> FPFHMethod::generateDescriptors(
        SpinImage::gpu::Mesh device_sceneAsMesh,
        SpinImage::gpu::PointCloud device_sceneAsPointCloud,
        SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::FPFHExecutionTimes fpfhExecutionTimes{};

    SpinImage::gpu::array<SpinImage::gpu::FPFHDescriptor> descriptors = SpinImage::gpu::generateFPFHHistograms(
            device_sceneAsPointCloud,
            device_descriptorOrigins,
            parameters.supportRadius,
            &fpfhExecutionTimes);

    executionTimes->append("total", fpfhExecutionTimes.totalExecutionTimeSeconds);
    executionTimes->append("reformat", fpfhExecutionTimes.originReformatExecutionTimeSeconds);
    executionTimes->append("spfh_origins", fpfhExecutionTimes.originSPFHGenerationExecutionTimeSeconds);
    executionTimes->append("spfh_pointCloud", fpfhExecutionTimes.pointCloudSPFHGenerationExecutionTimeSeconds);
    executionTimes->append("generation", fpfhExecutionTimes.fpfhGenerationExecutionTimeSeconds);

    return {descriptors.length, reinterpret_cast<char*>(descriptors.content)};
}

SpinImage::cpu::array<unsigned int> FPFHMethod::computeSearchResultRanks(
        SpinImage::gpu::array<char> device_needleDescriptors,
        SpinImage::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    SpinImage::debug::FPFHSearchExecutionTimes times{};

    SpinImage::cpu::array<unsigned int> searchResultIndices = SpinImage::gpu::computeFPFHSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<SpinImage::gpu::FPFHDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<SpinImage::gpu::FPFHDescriptor*>(device_haystackDescriptors.content)},
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

const std::string FPFHMethod::getMethodCommandLineParameterName() {
    return "fpfh";
}

const std::string FPFHMethod::getMethodDumpFileName() {
    return "FPFH";
}
