#include "clutterBoxKernels.cuh"

__global__ void scaleMesh() {
    /*double averageX = 0;
    double averageY = 0;
    double averageZ = 0;

    // I use a running average mean computing method here for better accuracy with large models
    for(unsigned int i = 0; i < input.vertexCount; i++) {
        float3_cpu vertex = input.vertices[i];

        averageX = ((float(i) * averageX) + vertex.x) / double(i + 1);
        averageY = ((float(i) * averageY) + vertex.y) / double(i + 1);
        averageZ = ((float(i) * averageZ) + vertex.z) / double(i + 1);
    }

    double maxDistance = std::numeric_limits<double>::max();

    for(unsigned int i = 0; i < input.vertexCount; i++) {
        float3_cpu vertex = input.vertices[i];

        double deltaX = vertex.x - averageX;
        double deltaY = vertex.y - averageY;
        double deltaZ = vertex.z - averageZ;

        double length = std::sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
        maxDistance = std::max(maxDistance, length);
    }

    HostMesh scaledMesh(input.vertexCount, input.indexCount);

    double scaleFactor = (1.0 / maxDistance) * radius;

    for(int i = 0; i < input.vertexCount; i++) {
        scaledMesh.vertices[i].x = float((double(input.vertices[i].x) - averageX) * scaleFactor);
        scaledMesh.vertices[i].y = float((double(input.vertices[i].y) - averageY) * scaleFactor);
        scaledMesh.vertices[i].z = float((double(input.vertices[i].z) - averageZ) * scaleFactor);
    }

    return scaledMesh;*/
}

void scaleMeshOnGPU(DeviceMesh mesh, float targetRadius) {

}