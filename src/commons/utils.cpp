#include "../../include/evo_alg/commons/utils.hpp"

namespace evo_alg {
    namespace utils {
        std::random_device rd;
        std::mt19937_64 rng(rd());

        int mpi_size = 1, mpi_rank = 1;

        void setSeedRNG(uint64_t seed) {
            rng.seed(seed);
        }

        double uniformProbGen() {
            std::uniform_real_distribution<double> uniform_dist(0, 1);
            return uniform_dist(rng);
        }

        void initMPI(int const size, int const rank) {
            mpi_size = size;
            mpi_rank = rank;
        }

        void Timer::startTimer(std::string timer_name) {
            start_time_[timer_name] = std::chrono::high_resolution_clock::now();
        }

        void Timer::stopTimer(std::string timer_name) {
            end_time_[timer_name] = std::chrono::high_resolution_clock::now();
        }

        double Timer::getTime(std::string timer_name) {
            std::chrono::duration<double, std::milli> time = end_time_[timer_name] - start_time_[timer_name];

            return time.count();
        }
    }
}