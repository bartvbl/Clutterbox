#include "QUICCIMethod.h"
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>

ShapeDescriptor::gpu::array<char> QUICCIMethod::generateDescriptors(
        ShapeDescriptor::gpu::Mesh device_sceneAsMesh,
        ShapeDescriptor::gpu::PointCloud device_sceneAsPointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
        Clutterbox::GenerationParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::QUICCIExecutionTimes quicciExecutionTimes{};

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::gpu::generateQUICCImages(
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

ShapeDescriptor::cpu::array<unsigned int> QUICCIMethod::computeSearchResultRanks(
        ShapeDescriptor::gpu::array<char> device_needleDescriptors,
        ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
        Clutterbox::SearchParameters parameters,
        ExecutionTimes *executionTimes) {

    ShapeDescriptor::debug::QUICCISearchExecutionTimes times{};

    ShapeDescriptor::cpu::array<unsigned int> searchResultIndices = ShapeDescriptor::gpu::computeQUICCImageSearchResultRanks(
            {device_needleDescriptors.length,
             reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(device_needleDescriptors.content)},
            {device_haystackDescriptors.length,
             reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(device_haystackDescriptors.content)},
             &times);

    executionTimes->append("total", times.totalExecutionTimeSeconds);
    executionTimes->append("search", times.searchExecutionTimeSeconds);

    return searchResultIndices;
}

void QUICCIMethod::dumpMetadata(json* jsonOutput) {
#if QUICCI_DISTANCE_FUNCTION == CLUTTER_RESISTANT_DISTANCE
    (*jsonOutput)["quicciDistanceFunction"] = "clutterResistant";
#elif QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
    (*jsonOutput)["quicciDistanceFunction"] = "weightedHamming";
#elif QUICCI_DISTANCE_FUNCTION == HAMMING_DISTANCE
    (*jsonOutput)["quicciDistanceFunction"] = "hamming";
#endif
    (*jsonOutput)["quicciImageWidthPixels"] = spinImageWidthPixels;
}

const std::string QUICCIMethod::getMethodCommandLineParameterName() {
    return "quicci";
}

const std::string QUICCIMethod::getMethodDumpFileName() {
    return "QUICCI";
}
