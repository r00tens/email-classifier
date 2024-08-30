#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

class Timer
{
public:
    Timer();
    ~Timer();

    void start();
    void stop();
    void reset();
    [[nodiscard]] double elapsed_time() const;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end_time;
    double m_elapsed_time{};
    bool m_running{};
};

#endif //TIMER_HPP
