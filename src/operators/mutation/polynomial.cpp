#include "../../../include/evo_alg/commons/macros.hpp"
#include "../../../include/evo_alg/commons/utils.hpp"
#include "../../../include/evo_alg/operators/mutation.hpp"

#include <iostream>

namespace evo_alg {
    namespace mutator {
        void polynomial(real_individual_t& individual, double const pr, uint32_t const n) {
            real_chromosome_t individual_chromosome = individual.getChromosome();
            real_chromosome_t mutated_chromosome = individual_chromosome;
            std::vector<std::pair<real_gene_t, real_gene_t>> bounds = individual.getBounds();
            for (size_t index = 0; index < mutated_chromosome.size(); ++index) {
                double const cur_pr = utils::uniformProbGen();
                if (cur_pr < pr) {
                    double const u = utils::uniformProbGen();
                    double const delta = u < 0.5 ? pow(2 * u, 1.0 / (n + 1)) - 1 : 1 - pow((2 * (1 - u)), 1 / (n + 1));
                    mutated_chromosome[index] +=
                        delta * (u < 0.5 ? individual_chromosome[index] - bounds[index].first
                                         : bounds[index].second - individual_chromosome[index]);
                }

                assert(utils::numericGreaterEqual(mutated_chromosome[index], bounds[index].first));
                assert(utils::numericLowerEqual(mutated_chromosome[index], bounds[index].second));
            }
            individual.setChromosome(mutated_chromosome);
        }
    }
}