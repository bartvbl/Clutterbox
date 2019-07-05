#include "clutterBoxKernels.cuh"
#include <iostream>

#define GLM_FORCE_CXX98
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "nvidia/helper_cuda.h"
#include "../../../../libShapeSearch/lib/nvidia-samples-common/nvidia/helper_cuda.h"
#include <cuda_runtime.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <nvidia/helper_cuda.h>

__host__ __device__ __inline__ size_t roundSizeToNearestCacheLine(size_t sizeInBytes) {
    return (sizeInBytes + 127u) & ~((size_t) 127);
}

__global__ void transformMeshes(glm::mat4* transformations, glm::mat3* normalMatrices, size_t* endIndices, DeviceMesh scene) {
    size_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex >= scene.vertexCount) {
        return;
    }

    unsigned int transformationIndex = 0;
    while(threadIndex >= endIndices[transformationIndex]) {
        transformationIndex++;
    }

    glm::vec4 vertex;
    vertex.x = scene.vertices_x[threadIndex];
    vertex.y = scene.vertices_y[threadIndex];
    vertex.z = scene.vertices_z[threadIndex];
    vertex.w = 1.0;

    glm::vec3 normal;
    normal.x = scene.normals_x[threadIndex];
    normal.y = scene.normals_y[threadIndex];
    normal.z = scene.normals_z[threadIndex];

    glm::vec4 transformedVertex = transformations[transformationIndex] * vertex;
    glm::vec3 transformedNormal = normalMatrices[transformationIndex] * normal;

    transformedNormal = glm::normalize(transformedNormal);

    scene.vertices_x[threadIndex] = transformedVertex.x;
    scene.vertices_y[threadIndex] = transformedVertex.y;
    scene.vertices_z[threadIndex] = transformedVertex.z;

    scene.normals_x[threadIndex] = transformedNormal.x;
    scene.normals_y[threadIndex] = transformedNormal.y;
    scene.normals_z[threadIndex] = transformedNormal.z;

}

void randomlyTransformMeshes(DeviceMesh scene, float maxDistance, std::vector<DeviceMesh> device_meshList, std::default_random_engine &randomGenerator) {
    std::vector<size_t> meshEndIndices(device_meshList.size());
    size_t currentEndIndex = 0;

    std::vector<glm::mat4> randomTransformations(device_meshList.size());
    std::vector<glm::mat3> randomNormalTransformations(device_meshList.size());

    std::uniform_real_distribution<float> distribution(0, 1);

    for(unsigned int i = 0; i < device_meshList.size(); i++) {
        float yaw = float(distribution(randomGenerator) * 2.0 * M_PI);
        float pitch = float((distribution(randomGenerator) - 0.5) * M_PI);
        float roll = float(distribution(randomGenerator) * 2.0 * M_PI);

        float distanceX = maxDistance * distribution(randomGenerator);
        float distanceY = maxDistance * distribution(randomGenerator);
        float distanceZ = maxDistance * distribution(randomGenerator);

        std::cout << "Rotation: (" << yaw << ", " << pitch << ", "<< roll << "), Translation: (" << distanceX << ", "<< distanceY << ", "<< distanceZ << "), Vertex Count: " << device_meshList.at(i).vertexCount << std::endl;

        glm::mat4 randomRotationTransformation(1.0);
        randomRotationTransformation = glm::rotate(randomRotationTransformation, yaw,   glm::vec3(0, 0, 1));
        randomRotationTransformation = glm::rotate(randomRotationTransformation, pitch, glm::vec3(0, 1, 0));
        randomRotationTransformation = glm::rotate(randomRotationTransformation, roll,  glm::vec3(1, 0, 0));

        glm::mat4 randomTransformation(1.0);
        randomTransformation = glm::translate(randomTransformation, glm::vec3(distanceX, distanceY, distanceZ));
        randomTransformation = randomTransformation * randomRotationTransformation;

        randomTransformations.at(i) = randomTransformation;
        randomNormalTransformations.at(i) = glm::mat3(randomRotationTransformation);

        currentEndIndex += device_meshList.at(i).vertexCount;
        meshEndIndices.at(i) = currentEndIndex;
    }

    glm::mat4* device_transformations;
    size_t transformationBufferSize = device_meshList.size() * sizeof(glm::mat4);
    checkCudaErrors(cudaMalloc(&device_transformations, transformationBufferSize));
    checkCudaErrors(cudaMemcpy(device_transformations, randomTransformations.data(), transformationBufferSize, cudaMemcpyHostToDevice));

    glm::mat3* device_normalMatrices;
    size_t normalMatrixBufferSize = device_meshList.size() * sizeof(glm::mat3);
    checkCudaErrors(cudaMalloc(&device_normalMatrices, normalMatrixBufferSize));
    checkCudaErrors(cudaMemcpy(device_normalMatrices, randomNormalTransformations.data(), normalMatrixBufferSize, cudaMemcpyHostToDevice));

    size_t* device_endIndices;
    size_t startIndexBufferSize = device_meshList.size() * sizeof(size_t);
    checkCudaErrors(cudaMalloc(&device_endIndices, startIndexBufferSize));
    checkCudaErrors(cudaMemcpy(device_endIndices, meshEndIndices.data(), startIndexBufferSize, cudaMemcpyHostToDevice));

    const size_t blockSize = 128;
    size_t blockCount = (scene.vertexCount / blockSize) + 1;
    transformMeshes<<<blockCount, blockSize>>>(device_transformations, device_normalMatrices, device_endIndices, scene);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    cudaFree(device_transformations);
    cudaFree(device_normalMatrices);
    cudaFree(device_endIndices);
}