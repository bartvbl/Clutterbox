#include "clutterKernel.cuh"
#include "../../../libShapeSearch/lib/nvidia-samples-common/nvidia/helper_cuda.h"
#include <nvidia/helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void computeClutterKernel(array<DeviceOrientedPoint> origins, SpinImage::GPUPointCloud samplePointCloud, float spinImageWidth, array<float> clutterValues, size_t referenceObjectSampleCount) {

    __shared__ size_t hitObjectSampleCount;
    __shared__ size_t hitClutterSampleCount;

    const size_t pointCloudSampleCount = samplePointCloud.vertices.length;

    if(threadIdx.x == 0) {
        hitSampleCount = 0;
        missSampleCount = 0;
    }

    __syncthreads();

    for(size_t sampleIndex = threadIdx.x; sampleIndex < pointCloudSampleCount; sampleIndex += blockDim.x) {

    }

    __syncthreads();

    if(threadIdx.x == 0) {
        double clutterPercentage = double(hitObjectSampleCount) / double(hitObjectSampleCount + hitClutterSampleCount);

        clutterValues.content[blockIdx.x] = float(clutterPercentage);
    }
}

array<float> computeClutter(array<DeviceOrientedPoint> origins, SpinImage::GPUPointCloud samplePointCloud, float spinImageWidth, size_t referenceObjectSampleCount) {
    array<float> device_clutterValues;

    size_t clutterBufferSize = origins.length * sizeof(float);
    checkCudaErrors(cudaMalloc(&device_clutterValues.content, clutterBufferSize));

    cudaMemset(device_clutterValues.content, 0, clutterBufferSize);

    computeClutterKernel<<<origins.length, 256>>>(origins, samplePointCloud, device_clutterValues);
    checkCudaErrors(cudaDeviceSynchronize());

    array<float> host_clutterValues;
    host_clutterValues.content = new float[origins.length];
    host_clutterValues.length = origins.length;

    checkCudaErrors(cudaMemcpy(host_clutterValues.content, device_clutterValues.content, clutterBufferSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(device_clutterValues.content));

    return host_clutterValues;

}