#include "FPFHMethod.h"
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramSearcher.cuh>

ShapeDescriptor::gpu::array<char> FPFHMethod::generateDescriptors(
        ShapeDescriptor::gpu::Mesh device_sceneAsMesh,
        ShapeDescriptor::gpu::PointCloud device_sceneAsPointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::FPFHExecutionTimes fpfhExecutionTimes{};

    ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::FPFHDescriptor> descriptors = ShapeDescriptor::gpu::generateFPFHHistograms(
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

ShapeDescriptor::cpu::array<unsigned int> FPFHMethod::computeSearchResultRanks(
        ShapeDescriptor::gpu::array<char> device_needleDescriptors,
        ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::FPFHSearchExecutionTimes times{};

    ShapeDescriptor::cpu::array<unsigned int> searchResultIndices = ShapeDescriptor::gpu::computeFPFHSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<ShapeDescriptor::gpu::FPFHDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<ShapeDescriptor::gpu::FPFHDescriptor*>(device_haystackDescriptors.content)},
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
