#ifndef GPUINFO_CUH
#define GPUINFO_CUH

#include <cuda_runtime.h>

class GPUInfo
{
public:
    GPUInfo();

    [[nodiscard]] auto getMaxThreadsPerBlock() const -> int;

    static auto calculateNumBlocks(int numElements, int blockSize) -> int;

private:
    int m_deviceId;
    cudaDeviceProp m_deviceProp;
};

#endif //GPUINFO_CUH
