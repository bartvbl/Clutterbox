#include "clutterBoxKernels.cuh"

__global__ void transformMeshes() {

}

void randomlyTransformMeshes(DeviceMesh scene, std::vector<DeviceMesh> device_meshList, std::default_random_engine randomGenerator) {
    std::vector<size_t> meshStartIndices(device_meshList.size());
    size_t currentStartIndex = 0;

    std::vector<glm::mat4> randomTransformations(device_meshList.size());

    for(unsigned int i = 0; i < device_meshList.size(); i++) {
        float yaw = float(randomGenerator() * 2.0 * M_PI);
        float pitch = float((randomGenerator() - 0.5) * M_PI);
        float roll = float(randomGenerator() * 2.0 * M_PI);

        glm::mat4 randomTransformation(1.0);
        randomTransformation = glm::rotate(randomTransformation, yaw,   glm::vec3(0, 0, 1));
        randomTransformation = glm::rotate(randomTransformation, pitch, glm::vec3(0, 1, 0));
        randomTransformation = glm::rotate(randomTransformation, roll,  glm::vec3(1, 0, 0));

        randomTransformations.at(i) = randomTransformation;

        meshStartIndices.at(i) = currentStartIndex;
        currentStartIndex += device_meshList.at(i).vertexCount;
    }

    glm::mat4* device_transformations;
    size_t transformationBufferSize = device_meshList.size() * sizeof(glm::mat4);
    checkCudaErrors(cudaMalloc(&device_transformations, transformationBufferSize));
    checkCudaErrors(cudaMemCpy(device_transformations, randomTransformations.data(), transformationBufferSize, cudaMemcpyHostToDevice));

    size_t* device_startIndices;
    size_t startIndexBufferSize = device_meshList.size() * sizeof(size_t);
    checkCudaErrors(cudaMalloc(&device_startIndices, startIndexBufferSize));
    checkCudaErrors(cudaMemCpy(device_startIndices, meshStartIndices.data(), startIndexBufferSize, cudaMemcpyHostToDevice));

    const size_t blockSize = 128;
    size_t blockCount = (scene.vertexCount / blockSize) + 1;
    transformMeshes<<<blockCount, blockSize>>>(device_transformations, device_startIndices, scene);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    cudaFree(device_transformations);
    cudaFree(device_startIndices);
}