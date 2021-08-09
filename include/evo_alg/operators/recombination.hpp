#ifndef GUARD_H_EVO_ALG_RECOMBINATION
#define GUARD_H_EVO_ALG_RECOMBINATION

#include "../../../include/evo_alg/core.hpp"

#include <utility>
#include <functional>

namespace evo_alg {
    namespace recombinator {
        template <class IndividualType>
        using crossover_function_t =
            std::function<void(IndividualType const&, IndividualType const&, IndividualType&, IndividualType&)>;

        void sbx(real_individual_t const& parent_1, real_individual_t const& parent_2, real_individual_t& child_1,
                 real_individual_t& child_2, uint32_t const n, double const pr);

        void arithmetic(real_individual_t const& parent_1, real_individual_t const& parent_2,
                        real_individual_t& child_1, real_individual_t& child_2);

        template <class GeneType>
        void eliteUniform(Individual<GeneType> const& elite_parent, Individual<GeneType> const& non_elite_parent,
                          Individual<GeneType>& child, double const elite_pr_lb) {
            size_t const chromosome_size = elite_parent.getBounds().size();

            std::vector<GeneType> elite_parent_chromosome = elite_parent.getChromosome();
            std::vector<GeneType> non_elite_parent_chromosome = non_elite_parent.getChromosome();
            std::vector<GeneType> child_chromosome = std::vector<GeneType>(chromosome_size);

            std::uniform_real_distribution<double> prob_gen(elite_pr_lb, 1.0 - 1.0 / chromosome_size);
            double const elite_pr = prob_gen(utils::rng);
            assert(elite_pr <= 1.0);
            for (size_t index = 0; index < chromosome_size; ++index) {
                double const pr = utils::uniformProbGen();
                child_chromosome[index] =
                    pr < elite_pr ? elite_parent_chromosome[index] : non_elite_parent_chromosome[index];
            }

            child.setChromosome(child_chromosome);
        }

        template <class GeneType>
        void onePoint(Individual<GeneType> const& parent_1, Individual<GeneType> const& parent_2,
                      Individual<GeneType>& child_1, Individual<GeneType>& child_2) {
            std::vector<GeneType> parent_1_chromosome = parent_1.getChromosome();
            std::vector<GeneType> parent_2_chromosome = parent_2.getChromosome();
            std::vector<GeneType> child_1_chromosome = parent_1_chromosome;
            std::vector<GeneType> child_2_chromosome = parent_2_chromosome;

            size_t const chromosome_size = parent_1_chromosome.size();
            std::uniform_int_distribution<size_t> point_dist(1, chromosome_size - 1);
            size_t const cut_point = point_dist(utils::rng);
            for (size_t index = cut_point; index < chromosome_size; ++index) {
                child_1_chromosome[index] = parent_2_chromosome[index];
                child_2_chromosome[index] = parent_1_chromosome[index];
            }
            child_1.setChromosome(child_1_chromosome);
            child_2.setChromosome(child_2_chromosome);
        }

        template <class GeneType>
        void twoPoints(Individual<GeneType> const& parent_1, Individual<GeneType> const& parent_2,
                       Individual<GeneType>& child_1, Individual<GeneType>& child_2) {
            std::vector<GeneType> parent_1_chromosome = parent_1.getChromosome();
            std::vector<GeneType> parent_2_chromosome = parent_2.getChromosome();
            std::vector<GeneType> child_1_chromosome = parent_1_chromosome;
            std::vector<GeneType> child_2_chromosome = parent_2_chromosome;

            size_t const chromosome_size = parent_1_chromosome.size();
            std::vector<size_t> indexes(chromosome_size);
            std::iota(indexes.begin(), indexes.end(), 0);
            std::shuffle(indexes.begin(), indexes.end(), utils::rng);

            size_t start_index = indexes[0];
            size_t end_index = indexes[1];

            if (start_index > end_index) {
                std::swap(start_index, end_index);
            }
            if (start_index == 0 && end_index == chromosome_size - 1) {
                end_index = indexes[2];
            }

            for (size_t index = start_index; index <= end_index; ++index) {
                child_1_chromosome[index] = parent_2_chromosome[index];
                child_2_chromosome[index] = parent_1_chromosome[index];
            }
            child_1.setChromosome(child_1_chromosome);
            child_2.setChromosome(child_2_chromosome);
        }
    }
}

#endif