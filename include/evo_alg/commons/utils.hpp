#ifndef GUARD_H_EVO_ALG_UTILS
#define GUARD_H_EVO_ALG_UTILS

#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <string>

namespace evo_alg {
    namespace utils {
        constexpr double eps = 1e-12;

        extern std::mt19937_64 rng;

        extern int mpi_size, mpi_rank;

        void setSeedRNG(uint64_t seed);

        double uniformProbGen();

        void initMPI(int const size, int const rank);

        template <typename T>
        bool numericLower(T const x, T const y, double const precision = eps) {
            return x - y < (T) -precision;
        }

        template <typename T>
        bool numericGreater(T const x, T const y, double const precision = eps) {
            return x - y > (T) precision;
        }

        template <typename T>
        bool numericEqual(T const x, T const y, double const precision = eps) {
            return x - y <= (T) precision && x - y >= (T) -precision;
        }

        template <typename T>
        bool numericLowerEqual(T const x, T const y, double const precision = eps) {
            return numericLower(x, y, precision) || numericEqual(x, y, precision);
        }

        template <typename T>
        bool numericGreaterEqual(T const x, T const y, double const precision = eps) {
            return numericGreater(x, y, precision) || numericEqual(x, y, precision);
        }

        class Timer {
          public:
            void startTimer(std::string timer_name);
            void stopTimer(std::string timer_name);
            double getTime(std::string timer_name);

          private:
            std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> start_time_;
            std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> end_time_;
        };
    }
}

#endif