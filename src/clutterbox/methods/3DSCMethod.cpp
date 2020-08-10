#include "3DSCMethod.h"
#include <shapeDescriptor/gpu/3dShapeContextGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>

ShapeDescriptor::gpu::array<char> SCMethod::generateDescriptors(
        ShapeDescriptor::gpu::Mesh device_sceneAsMesh,
        ShapeDescriptor::gpu::PointCloud device_sceneAsPointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::SCExecutionTimes scExecutionTimes{};

    ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptors = ShapeDescriptor::gpu::generate3DSCDescriptors(
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

ShapeDescriptor::cpu::array<unsigned int> SCMethod::computeSearchResultRanks(
        ShapeDescriptor::gpu::array<char> device_needleDescriptors,
        ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::SCSearchExecutionTimes times{};

    ShapeDescriptor::cpu::array<unsigned int> searchResultIndices = ShapeDescriptor::gpu::compute3DSCSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<ShapeDescriptor::ShapeContextDescriptor*>(device_needleDescriptors.content)},
             parameters.needleDescriptorScenePointCloudPointCount,
            {device_haystackDescriptors.length,
             reinterpret_cast<ShapeDescriptor::ShapeContextDescriptor*>(device_haystackDescriptors.content)},
             parameters.haystackDescriptorScenePointCloudPointCount,
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

void SCMethod::dumpMetadata(json jsonOutput) {
    jsonOutput["3dscMinSupportRadius"] = minSupportRadius;
    jsonOutput["3dscPointDensityRadius"] = pointDensityRadius;
}

const std::string SCMethod::getMethodCommandLineParameterName() {
    return "3dsc";
}

const std::string SCMethod::getMethodDumpFileName() {
    return "3DSC";
}

