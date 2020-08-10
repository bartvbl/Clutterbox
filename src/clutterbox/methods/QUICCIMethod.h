#pragma once

#include <clutterbox/methods/types/ClutterboxMethod.h>

class QUICCIMethod : public Clutterbox::Method {
    ShapeDescriptor::gpu::array<char>
    generateDescriptors(ShapeDescriptor::gpu::Mesh device_sceneAsMesh,
                        ShapeDescriptor::gpu::PointCloud device_sceneAsPointCloud,
                        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> device_origins,
                        Clutterbox::GenerationParameters parameters,
                        ExecutionTimes *executionTimes) override;

    ShapeDescriptor::cpu::array<unsigned int> computeSearchResultRanks(ShapeDescriptor::gpu::array<char> device_needleDescriptors,
                                                            ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
                                                            Clutterbox::SearchParameters parameters,
                                                            ExecutionTimes *executionTimes) override;

    const std::string getMethodCommandLineParameterName() override;

    const std::string getMethodDumpFileName() override;
};