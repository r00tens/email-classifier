#include "Timer.hpp"

Timer::Timer() = default;

Timer::~Timer() = default;

void Timer::start()
{
    if (!m_running)
    {
        m_start_time = std::chrono::high_resolution_clock::now();
        m_running = true;
    }
}

void Timer::stop()
{
    if (m_running)
    {
        m_end_time = std::chrono::high_resolution_clock::now();
        m_elapsed_time = std::chrono::duration<double>(m_end_time - m_start_time).count();
        m_running = false;
    }
}

void Timer::reset()
{
    m_elapsed_time = 0.0;
    m_running = false;
}

double Timer::elapsed_time() const
{
    return m_elapsed_time;
}
