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


__global__ void detectDuplicates(DeviceMesh mesh, bool* isDuplicate) {
    size_t vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(vertexIndex >= mesh.vertexCount) {
        return;
    }

    float3 vertex = make_float3(
            mesh.vertices_x[vertexIndex],
            mesh.vertices_y[vertexIndex],
            mesh.vertices_z[vertexIndex]);
    float3 normal = make_float3(
            mesh.normals_x[vertexIndex],
            mesh.normals_y[vertexIndex],
            mesh.normals_z[vertexIndex]);

    for(size_t i = 0; i < vertexIndex; i++) {
        float3 otherVertex = make_float3(
                mesh.vertices_x[i],
                mesh.vertices_y[i],
                mesh.vertices_z[i]);
        float3 otherNormal = make_float3(
                mesh.normals_x[i],
                mesh.normals_y[i],
                mesh.normals_z[i]);

        // We're looking for exact matches here. Given that vertex duplications should
        // yield equivalent vertex coordinates, testing floating point numbers for
        // exact equivalence is warranted.
        if( vertex.x == otherVertex.x &&
            vertex.y == otherVertex.y &&
            vertex.z == otherVertex.z &&
            normal.x == otherNormal.x &&
            normal.y == otherNormal.y &&
            normal.z == otherNormal.z) {

            isDuplicate[vertexIndex] = true;
            return;
        }
    }

    isDuplicate[vertexIndex] = false;
}

__global__ void computeTargetIndices(array<signed long long> targetIndices, bool* duplicateVertices, size_t vertexCount) {
    size_t vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(vertexIndex >= vertexCount) {
        return;
    }

    // The value of -1 indicates that the vertex is a duplicate of another one
    // and should therefore be discarded
    signed long long targetIndex = -1;

    bool isDuplicate = duplicateVertices[vertexIndex];

    if(!isDuplicate) {
        for(size_t i = 0; i < vertexIndex; i++) {
            // If it is a duplicate, it will get removed
            // Otherwise, it'll be added in front of the current entry
            targetIndex += duplicateVertices[i] ? 0 : 1;
        }
    }

    targetIndices.content[vertexIndex] = targetIndex;
}

array<signed long long> computeUniqueIndexMapping(DeviceMesh boxScene, std::vector<DeviceMesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts, size_t &totalUniqueVertexCount) {
    size_t sceneVertexCount = boxScene.vertexCount;

    bool* device_duplicateVertices;
    checkCudaErrors(cudaMalloc(&device_duplicateVertices, sceneVertexCount * sizeof(bool)));
    detectDuplicates<<<(boxScene.vertexCount / 256) + 1, 256>>>(boxScene, device_duplicateVertices);
    checkCudaErrors(cudaDeviceSynchronize());

    bool* temp_duplicateVertices = new bool[sceneVertexCount];
    checkCudaErrors(cudaMemcpy(temp_duplicateVertices, device_duplicateVertices, boxScene.vertexCount * sizeof(bool), cudaMemcpyDeviceToHost));

    size_t baseIndex = 0;
    totalUniqueVertexCount = 0;
    for(auto mesh : deviceMeshes) {
        size_t meshUniqueVertexCount = 0;
        for(size_t i = 0; i < mesh.vertexCount; i++) {
            // Check if the vertex is unique
            if(temp_duplicateVertices[baseIndex + i] == false) {
                totalUniqueVertexCount++;
                meshUniqueVertexCount++;
            }
        }
        baseIndex += meshUniqueVertexCount;
        uniqueVertexCounts->push_back(meshUniqueVertexCount);
    }

    delete[] temp_duplicateVertices;

    array<signed long long> device_uniqueIndexMapping;
    device_uniqueIndexMapping.length = boxScene.vertexCount;
    checkCudaErrors(cudaMalloc(&device_uniqueIndexMapping.content, boxScene.vertexCount * sizeof(signed long long)));
    computeTargetIndices<<<(boxScene.vertexCount / 256) + 1, 256>>>(device_uniqueIndexMapping, device_duplicateVertices, boxScene.vertexCount);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(device_duplicateVertices));

    return device_uniqueIndexMapping;
}

__global__ void mapVertices(DeviceMesh boxScene, array<DeviceOrientedPoint> origins, array<signed long long> mapping) {
    size_t vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(vertexIndex >= boxScene.vertexCount) {
        return;
    }

    signed long long targetIndex = mapping.content[vertexIndex];

    if(targetIndex != -1) {
        float3 vertex = make_float3(
                boxScene.vertices_x[vertexIndex],
                boxScene.vertices_y[vertexIndex],
                boxScene.vertices_z[vertexIndex]);
        float3 normal = make_float3(
                boxScene.normals_x[vertexIndex],
                boxScene.normals_y[vertexIndex],
                boxScene.normals_z[vertexIndex]);

        DeviceOrientedPoint origin;
        origin.vertex = vertex;
        origin.normal = normal;

        origins.content[targetIndex] = origin;
    }
}

array<DeviceOrientedPoint> applyUniqueMapping(DeviceMesh boxScene, array<signed long long> device_mapping, size_t totalUniqueVertexCount) {
    assert(boxScene.vertexCount == device_mapping.length);

    array<DeviceOrientedPoint> device_origins;
    device_origins.length = totalUniqueVertexCount;
    checkCudaErrors(cudaMalloc(&device_origins.content, totalUniqueVertexCount * sizeof(DeviceOrientedPoint)));

    mapVertices<<<(boxScene.vertexCount / 256) + 1, 256>>>(boxScene, device_origins, device_mapping);
    checkCudaErrors(cudaDeviceSynchronize());

    return device_origins;
}

array<DeviceOrientedPoint> computeUniqueSpinOrigins(DeviceMesh &mesh) {
    std::vector<DeviceMesh> deviceMeshes;
    deviceMeshes.push_back(mesh);
    std::vector<size_t> vertexCounts;
    size_t totalUniqueVertexCount;
    array<signed long long> device_mapping = computeUniqueIndexMapping(mesh, deviceMeshes, &vertexCounts, totalUniqueVertexCount);
    array<DeviceOrientedPoint> device_origins = applyUniqueMapping(mesh, device_mapping, totalUniqueVertexCount);
    checkCudaErrors(cudaFree(device_mapping.content));
    return device_origins;
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

        std::cout << "\t\tRotation: (" << yaw << ", " << pitch << ", "<< roll << "), Translation: (" << distanceX << ", "<< distanceY << ", "<< distanceZ << "), Vertex Count: " << device_meshList.at(i).vertexCount << std::endl;

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