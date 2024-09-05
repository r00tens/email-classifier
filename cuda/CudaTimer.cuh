#ifndef CUDATIMER_CUH
#define CUDATIMER_CUH

#include <cuda_runtime.h>

class CudaTimer
{
public:
    CudaTimer();
    ~CudaTimer();

    CudaTimer(const CudaTimer&) = delete;
    auto operator=(const CudaTimer&) -> CudaTimer& = delete;

    CudaTimer(CudaTimer&& other) = delete;
    auto operator=(CudaTimer&& other) -> CudaTimer& = delete;

    void start();
    void stop();
    void reset();
    void resume();

    [[nodiscard]] auto getTimeInSeconds() const -> float;

private:
    cudaEvent_t m_startEvent{};
    cudaEvent_t m_stopEvent{};
    float m_elapsedTime;
    bool m_running;

    static constexpr float MILLISECONDS_TO_SECONDS = 1000.0F;
};

#endif // CUDATIMER_CUH
