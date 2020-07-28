#pragma once

#include <clutterbox/methods/types/ClutterboxMethod.h>

class SCMethod : ClutterboxMethod {
    float minSupportRadius;
    float pointDensityRadius;

    SpinImage::array<char>
    generateDescriptors(SpinImage::gpu::Mesh device_sceneMesh,
                        SpinImage::gpu::PointCloud device_scenePointCloud,
                        SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_origins,
                        float supportRadius,
                        ExecutionTimes *executionTimes) override;

    SpinImage::array<unsigned int> computeSearchResultRanks(SpinImage::array<char> device_needleDescriptors,
                                                            SpinImage::array<char> device_haystackDescriptors,
                                                            ExecutionTimes *executionTimes) override;

    const std::string getMethodCommandLineParameterName() override;

    const std::string getMethodDumpFileName() override;

    explicit SCMethod(float minSupportRadius, float pointDensityRadius)
     : minSupportRadius(minSupportRadius),
       pointDensityRadius(pointDensityRadius) {}
};