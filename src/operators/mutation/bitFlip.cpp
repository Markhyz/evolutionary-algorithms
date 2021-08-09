#include "../../../include/evo_alg/commons/utils.hpp"
#include "../../../include/evo_alg/operators/mutation.hpp"

namespace evo_alg {
    namespace mutator {
        void bitFlip(binary_individual_t& individual, double const pr) {
            binary_chromosome_t chromosome = individual.getChromosome();
            for (size_t index = 0; index < chromosome.size(); ++index) {
                double const cur_pr = utils::uniformProbGen();
                if (cur_pr < pr) {
                    chromosome[index] = !chromosome[index];
                }
            }
            individual.setChromosome(chromosome);
        }
    }
}