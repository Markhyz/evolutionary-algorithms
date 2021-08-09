#include "../../../include/evo_alg/commons/utils.hpp"
#include "../../../include/evo_alg/operators/selection.hpp"

namespace evo_alg {
    namespace selector {
        size_t tournament(std::vector<double> const& individuals_fit, size_t const size) {
            std::vector<size_t> indexes(individuals_fit.size());
            std::iota(indexes.begin(), indexes.end(), 0);
            std::shuffle(indexes.begin(), indexes.end(), utils::rng);

            size_t best_individual = indexes[0];
            for (size_t index = 1; index < size; ++index)
                if (individuals_fit[indexes[index]] > individuals_fit[best_individual])
                    best_individual = indexes[index];

            return best_individual;
        }
    }
}