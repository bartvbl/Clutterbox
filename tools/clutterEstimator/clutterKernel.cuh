#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/GPUPointCloud.h>

array<float> computeClutter(array<DeviceOrientedPoint> array, SpinImage::GPUPointCloud cloud, float spinImageWidth, size_t referenceObjectSampleCount);