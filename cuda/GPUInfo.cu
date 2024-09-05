#include "GPUInfo.cuh"

GPUInfo::GPUInfo()
    : m_deviceId(0), m_deviceProp()
{
    cudaGetDevice(&m_deviceId);
    cudaGetDeviceProperties(&m_deviceProp, m_deviceId);
}

auto GPUInfo::getMaxThreadsPerBlock() const -> int
{
    return m_deviceProp.maxThreadsPerBlock;
}

auto GPUInfo::calculateNumBlocks(const int numElements, const int blockSize) -> int
{
    return (numElements + blockSize - 1) / blockSize;
}
