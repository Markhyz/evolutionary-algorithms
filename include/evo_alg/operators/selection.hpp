#ifndef GUARD_H_EVO_ALG_SELECTION
#define GUARD_H_EVO_ALG_SELECTION

#include "../../../include/evo_alg/commons/utils.hpp"
#include "../../../include/evo_alg/core.hpp"

#include <functional>

namespace evo_alg {
    namespace selector {
        template <class IndividualType>
        using selection_function_t = std::function<size_t(std::vector<double> const&)>;

        size_t tournament(std::vector<double> const& individuals_fit, size_t const size);
        size_t roulette(std::vector<double> const& individuals_fit);
    }
}

#endif