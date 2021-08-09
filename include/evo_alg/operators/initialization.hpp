#ifndef GUARD_H_EVO_ALG_INITIALIZATION
#define GUARD_H_EVO_ALG_INITIALIZATION

#include "../../../include/evo_alg/commons/utils.hpp"
#include "../../../include/evo_alg/core.hpp"

#include <functional>

namespace evo_alg {
    namespace initializator {
        template <class IndividualType, class FitnessType>
        using initialization_function_t =
            std::function<void(Population<IndividualType>&, typename FitnessType::const_shared_ptr, size_t const)>;

        template <typename GeneType>
        void uniformRandomInit(Population<Individual<GeneType>>& population,
                               typename FitnessFunction<GeneType>::const_shared_ptr fitness, size_t const size) {
            std::vector<std::pair<GeneType, GeneType>> const bounds = fitness->getBounds();
            size_t const chromosome_size = bounds.size();
            for (size_t index = 0; index < size; ++index) {
                std::vector<GeneType> new_chromosome(chromosome_size);

                for (size_t gene_index = 0; gene_index < chromosome_size; ++gene_index) {
                    if constexpr (std::is_integral_v<GeneType>) {
                        std::uniform_int_distribution<
                            std::conditional_t<std::is_same_v<GeneType, bool>, short, GeneType>>
                            geneGenerator(bounds[gene_index].first, bounds[gene_index].second);
                        new_chromosome[gene_index] = geneGenerator(utils::rng);
                    } else {
                        std::uniform_real_distribution<GeneType> geneGenerator(bounds[gene_index].first,
                                                                               bounds[gene_index].second);
                        new_chromosome[gene_index] = geneGenerator(utils::rng);
                    }
                }

                population[index].setChromosome(new_chromosome);
            }
        }
    }
}

#endif