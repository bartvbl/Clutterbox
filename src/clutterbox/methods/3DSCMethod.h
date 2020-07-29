#pragma once

#include <clutterbox/methods/types/ClutterboxMethod.h>

class SCMethod : public Clutterbox::Method {
    float minSupportRadius;
    float pointDensityRadius;

    SpinImage::array<char>
    generateDescriptors(SpinImage::gpu::Mesh device_sceneAsMesh,
                        SpinImage::gpu::PointCloud device_sceneAsPointCloud,
                        SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_origins,
                        Clutterbox::GenerationParameters parameters,
                        ExecutionTimes *executionTimes) override;

    SpinImage::array<unsigned int> computeSearchResultRanks(SpinImage::array<char> device_needleDescriptors,
                                                            SpinImage::array<char> device_haystackDescriptors,
                                                            Clutterbox::SearchParameters parameters,
                                                            ExecutionTimes *executionTimes) override;

    const std::string getMethodCommandLineParameterName() override;

    const std::string getMethodDumpFileName() override;

public:
    explicit SCMethod(float minSupportRadius, float pointDensityRadius)
     : minSupportRadius(minSupportRadius),
       pointDensityRadius(pointDensityRadius) {}
};