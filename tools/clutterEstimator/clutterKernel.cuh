#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/GPUPointCloud.h>
#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/utilities/meshSampler.cuh>
#include <vector>

array<float> computeClutter(
        array<DeviceOrientedPoint> array,
        SpinImage::GPUPointCloud cloud,
        float spinImageWidth,
        size_t referenceObjectSampleCount,
        size_t referenceObjectOriginCount);
size_t computeReferenceSampleCount(
        DeviceMesh referenceMesh,
        size_t totalSceneSampleCount,
        array<float> cumulativeAreaArray);