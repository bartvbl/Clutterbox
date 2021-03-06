#pragma once

#include <clutterbox/methods/types/ClutterboxMethod.h>

class SIMethod : public Clutterbox::Method {
    float supportAngle;

    ShapeDescriptor::gpu::array<char>
    generateDescriptors(ShapeDescriptor::gpu::Mesh device_sceneAsMesh,
                        ShapeDescriptor::gpu::PointCloud device_sceneAsPointCloud,
                        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_origins,
                        Clutterbox::GenerationParameters parameters,
                        ExecutionTimes *executionTimes) override;

    ShapeDescriptor::cpu::array<unsigned int> computeSearchResultRanks(
            ShapeDescriptor::gpu::array<char> device_needleDescriptors,
            ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
            Clutterbox::SearchParameters parameters,
            ExecutionTimes *executionTimes) override;

    void dumpMetadata(json* jsonOutput) override;

    const std::string getMethodCommandLineParameterName() override;

    const std::string getMethodDumpFileName() override;

public:
    explicit SIMethod(float supportAngle) : supportAngle(supportAngle) {}
};