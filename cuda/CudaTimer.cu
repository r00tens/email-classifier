#include "CudaTimer.cuh"

#include <stdexcept>

CudaTimer::CudaTimer() : m_elapsedTime(0), m_running(false)
{
    cudaEventCreate(&m_startEvent);
    cudaEventCreate(&m_stopEvent);
}

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(m_startEvent);
    cudaEventDestroy(m_stopEvent);
}

void CudaTimer::start()
{
    if (!m_running)
    {
        // Record the start event.
        cudaEventRecord(m_startEvent, nullptr);
        m_running = true;
    }
    else
    {
        throw std::runtime_error("Timer is already running!");
    }
}

void CudaTimer::stop()
{
    if (m_running)
    {
        // Record the stop event.
        cudaEventRecord(m_stopEvent, nullptr);
        cudaEventSynchronize(m_stopEvent); // Ensure events are synchronized.

        float time = 0.0F;
        cudaEventElapsedTime(&time, m_startEvent, m_stopEvent);

        m_elapsedTime += time; // Add the elapsed time to the total time.
        m_running = false;
    }
    else
    {
        throw std::runtime_error("Timer is not running!");
    }
}

void CudaTimer::reset()
{
    // Reset the stored elapsed time.
    m_elapsedTime = 0.0F;
    m_running = false;
}

void CudaTimer::resume()
{
    if (!m_running)
    {
        // Resume by recording the start event again.
        cudaEventRecord(m_startEvent, nullptr);
        m_running = true;
    }
    else
    {
        throw std::runtime_error("Timer is already running!");
    }
}

auto CudaTimer::getTimeInSeconds() const -> float
{
    return m_elapsedTime / MILLISECONDS_TO_SECONDS;
}
