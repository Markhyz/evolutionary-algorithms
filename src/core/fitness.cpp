#include "../../include/evo_alg/core/fitness.hpp"
#include "../../include/evo_alg/commons/utils.hpp"

#include <queue>

namespace evo_alg {
    namespace fitness {
        std::vector<double> linearScale(std::vector<double> const& f, double const c) {
            std::vector<double> sf(f.size());
            double f_min = 1e9, f_avg = 0, f_max = -1e9, f_tot = 0;
            for (size_t i = 0; i < f.size(); ++i) {
                f_min = std::min(f_min, f[i]);
                f_max = std::max(f_max, f[i]);
                f_tot += f[i];
            }
            f_avg = f_tot / (double) f.size();
            double alfa, beta;
            if (f_min > (c * f_avg - f_max) / (c - 1)) {
                if (utils::numericEqual(f_max, f_avg)) {
                    sf = f;
                    return sf;
                }
                alfa = (f_avg * (c - 1)) / (f_max - f_avg);
                beta = (f_avg * (f_max - c * f_avg)) / (f_max - f_avg);
            } else {
                if (utils::numericEqual(f_avg, f_min)) {
                    sf = f;
                    return sf;
                }
                alfa = f_avg / (f_avg - f_min);
                beta = (-f_min * f_avg) / (f_avg - f_min);
            }

            for (size_t i = 0; i < f.size(); ++i) {
                sf[i] = f[i] * alfa + beta;
            }

            return sf;
        }

        std::vector<double> linearNormalization(std::vector<double> const& fitness_values, double const min_value,
                                                double const max_value) {
            if (utils::numericEqual(min_value, max_value))
                return fitness_values;

            std::vector<double> result(fitness_values);
            for (double& fit : result) {
                fit = (fit - min_value + utils::eps) / (max_value - min_value + utils::eps);
            }

            return result;
        }

        std::vector<double> crowdingDistance(std::vector<std::vector<size_t>> frontiers,
                                             std::vector<FitnessValue> fitness_values) {
            std::vector<double> crowding_distance(fitness_values.size());
            size_t fitness_dimension = fitness_values[0].getDimension();
            for (std::vector<size_t>& frontier : frontiers) {
                for (size_t fit_index = 0; fit_index < fitness_dimension; ++fit_index) {
                    sort(frontier.begin(), frontier.end(), [&fitness_values, &fit_index](size_t ind_1, size_t ind_2) {
                        return fitness_values[ind_1][fit_index] < fitness_values[ind_2][fit_index];
                    });
                    double min_value = fitness_values[frontier.front()][fit_index];
                    double max_value = fitness_values[frontier.back()][fit_index];
                    crowding_distance[frontier.front()] += 1e9;
                    crowding_distance[frontier.back()] += 1e9;
                    for (size_t ind_index = 1; ind_index < frontier.size() - 1; ++ind_index) {
                        crowding_distance[frontier[ind_index]] +=
                            (fitness_values[frontier[ind_index + 1]][fit_index] -
                             fitness_values[frontier[ind_index - 1]][fit_index] + utils::eps) /
                            (max_value - min_value + utils::eps);

                        if (!utils::numericGreaterEqual(crowding_distance[frontier[ind_index]], 0.0)) {
                            std::cout << "Crowding error" << std::endl;
                            std::cout << crowding_distance[frontier[ind_index]] << std::endl;
                            std::cout << max_value << " " << min_value << " "
                                      << fitness_values[frontier[ind_index + 1]][fit_index] << " "
                                      << fitness_values[frontier[ind_index - 1]][fit_index] << std::endl;
                            exit(-1);
                        }
                    }
                }
            }

            return crowding_distance;
        }

        std::vector<std::vector<size_t>> nonDominatedSorting(std::vector<FitnessValue> const& fitness_values,
                                                             bool const crowding_sort) {
            std::vector<std::vector<size_t>> frontiers;

            std::vector<size_t> dominance_counter(fitness_values.size());
            std::vector<std::vector<size_t>> dominance_list(fitness_values.size());
            for (size_t ind_1 = 0; ind_1 < fitness_values.size(); ++ind_1) {
                for (size_t ind_2 = ind_1 + 1; ind_2 < fitness_values.size(); ++ind_2) {
                    if (fitness_values[ind_1] > fitness_values[ind_2]) {
                        ++dominance_counter[ind_2];
                        dominance_list[ind_1].push_back(ind_2);
                    }
                    if (fitness_values[ind_2] > fitness_values[ind_1]) {
                        ++dominance_counter[ind_1];
                        dominance_list[ind_2].push_back(ind_1);
                    }
                }
            }

            std::vector<size_t> next_frontier;
            std::queue<size_t> ind_queue;
            size_t current_frontier_size = 0;
            for (size_t ind = 0; ind < fitness_values.size(); ++ind) {
                if (dominance_counter[ind] == 0) {
                    next_frontier.push_back(ind);
                    ind_queue.push(ind);
                }
            }
            frontiers.push_back(next_frontier);
            current_frontier_size = next_frontier.size();
            next_frontier.clear();

            assert(ind_queue.size() > 0);
            while (ind_queue.size() > 0) {
                --current_frontier_size;

                size_t current_ind = ind_queue.front();
                ind_queue.pop();
                for (size_t ind : dominance_list[current_ind]) {
                    --dominance_counter[ind];
                    if (dominance_counter[ind] == 0) {
                        ind_queue.push(ind);
                        next_frontier.push_back(ind);
                    }
                }

                if (current_frontier_size == 0 && !next_frontier.empty()) {
                    frontiers.push_back(next_frontier);
                    current_frontier_size = next_frontier.size();
                    next_frontier.clear();
                }
            }

            if (crowding_sort) {
                std::vector<double> const crowding_distance = crowdingDistance(frontiers, fitness_values);
                for (std::vector<size_t>& frontier : frontiers) {
                    sort(frontier.rbegin(), frontier.rend(), [&crowding_distance](size_t ind_1, size_t ind_2) {
                        return crowding_distance[ind_1] < crowding_distance[ind_2];
                    });
                }
            }

            return frontiers;
        }

        FitnessValue::FitnessValue(){};
        FitnessValue::FitnessValue(std::vector<double> values) : fitness_values_(values){};
        FitnessValue::FitnessValue(std::initializer_list<double> values) : fitness_values_(values){};

        std::vector<double> const& FitnessValue::getValues() const {
            return fitness_values_;
        }

        size_t FitnessValue::getDimension() const {
            return fitness_values_.size();
        }

        double& FitnessValue::operator[](size_t const index) {
            return fitness_values_[index];
        }

        double FitnessValue::operator[](size_t const index) const {
            return fitness_values_[index];
        }

        bool FitnessValue::operator==(FitnessValue const& target_fit) const {
            std::vector<double> const& target_fit_values = target_fit.fitness_values_;

            for (size_t index = 0; index < fitness_values_.size(); ++index) {
                if (!utils::numericEqual(fitness_values_[index], target_fit_values[index])) {
                    return false;
                }
            }

            return true;
        }

        bool FitnessValue::operator!=(FitnessValue const& target_fit) const {
            return !(*this == target_fit);
        }

        bool FitnessValue::operator<(FitnessValue const& target_fit) const {
            std::vector<double> const& target_fit_values = target_fit.fitness_values_;

            bool equal = true;
            for (size_t index = 0; index < fitness_values_.size(); ++index) {
                if (utils::numericGreater(fitness_values_[index], target_fit_values[index])) {
                    return false;
                }
                if (utils::numericLower(fitness_values_[index], target_fit_values[index])) {
                    equal = false;
                }
            }

            return !equal;
        }

        bool FitnessValue::operator>(FitnessValue const& target_fit) const {
            return target_fit < *this;
        }

        bool FitnessValue::operator<=(FitnessValue const& target_fit) const {
            return *this < target_fit || *this == target_fit;
        }

        bool FitnessValue::operator>=(FitnessValue const& target_fit) const {
            return *this > target_fit || *this == target_fit;
        }
    }
}