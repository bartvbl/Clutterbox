#pragma once

#include <clutterbox/methods/types/ClutterboxMethod.h>

class RICIMethod : public Clutterbox::Method {
    SpinImage::gpu::array<char>
    generateDescriptors(SpinImage::gpu::Mesh device_sceneAsMesh,
                        SpinImage::gpu::PointCloud device_sceneAsPointCloud,
                        SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_origins,
                        Clutterbox::GenerationParameters parameters,
                        ExecutionTimes *executionTimes) override;

    SpinImage::cpu::array<unsigned int> computeSearchResultRanks(SpinImage::gpu::array<char> device_needleDescriptors,
                                                            SpinImage::gpu::array<char> device_haystackDescriptors,
                                                            Clutterbox::SearchParameters parameters,
                                                            ExecutionTimes *executionTimes) override;

    const std::string getMethodCommandLineParameterName() override;

    const std::string getMethodDumpFileName() override;
};