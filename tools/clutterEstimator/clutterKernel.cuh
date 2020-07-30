#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/utilities/kernels/meshSampler.cuh>
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