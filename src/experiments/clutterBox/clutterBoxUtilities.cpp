#include <cuda_runtime.h>
#include "clutterBoxUtilities.h"

DeviceMesh combineMeshesOnGPU(std::vector<DeviceMesh> meshes) {
    unsigned int totalVertexCount = 0;
    for(unsigned int i = 0; i < meshes.size(); i++) {
        totalVertexCount += meshes.at(i).vertexCount;
    }

    DeviceMesh combinedMesh;

    cudaMalloc(&combinedMesh.normals_x, totalVertexCount * sizeof(float));
    cudaMalloc(&combinedMesh.normals_y, totalVertexCount * sizeof(float));
    cudaMalloc(&combinedMesh.normals_z, totalVertexCount * sizeof(float));

    cudaMalloc(&combinedMesh.vertices_x, totalVertexCount * sizeof(float));
    cudaMalloc(&combinedMesh.vertices_y, totalVertexCount * sizeof(float));
    cudaMalloc(&combinedMesh.vertices_z, totalVertexCount * sizeof(float));

    combinedMesh.vertexCount = totalVertexCount;

    unsigned int pointerBaseIndex = 0;
    for(unsigned int i = 0; i < meshes.size(); i++) {
        cudaMemcpy(combinedMesh.normals_x + pointerBaseIndex, meshes.at(i).normals_x, meshes.at(i).vertexCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(combinedMesh.normals_y + pointerBaseIndex, meshes.at(i).normals_y, meshes.at(i).vertexCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(combinedMesh.normals_z + pointerBaseIndex, meshes.at(i).normals_z, meshes.at(i).vertexCount * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaMemcpy(combinedMesh.vertices_x + pointerBaseIndex, meshes.at(i).vertices_x, meshes.at(i).vertexCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(combinedMesh.vertices_y + pointerBaseIndex, meshes.at(i).vertices_y, meshes.at(i).vertexCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(combinedMesh.vertices_z + pointerBaseIndex, meshes.at(i).vertices_z, meshes.at(i).vertexCount * sizeof(float), cudaMemcpyDeviceToDevice);

        pointerBaseIndex += meshes.at(i).vertexCount;
    }

    return combinedMesh;
}