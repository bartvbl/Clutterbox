#pragma once

#include <spinImage/gpu/types/array.h>
#include <spinImage/cpu/types/array.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/utilities/kernels/meshSampler.cuh>
#include <vector>

SpinImage::cpu::array<float> computeClutter(
        SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> array,
        SpinImage::gpu::PointCloud cloud,
        float spinImageWidth,
        size_t referenceObjectSampleCount,
        size_t referenceObjectOriginCount);

size_t computeReferenceSampleCount(
        SpinImage::gpu::Mesh referenceMesh,
        size_t totalSceneSampleCount,
        SpinImage::gpu::array<float> cumulativeAreaArray);