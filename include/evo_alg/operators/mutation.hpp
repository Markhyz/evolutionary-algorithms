#ifndef GUARD_H_EVO_ALG_MUTATION
#define GUARD_H_EVO_ALG_MUTATION

#include "../../../include/evo_alg/core.hpp"

#include <functional>

namespace evo_alg {
    namespace mutator {
        template <class IndividualType>
        using mutation_function_t = std::function<void(IndividualType&, double const)>;

        void polynomial(real_individual_t& individual, double const pr, uint32_t const n);
        void bitFlip(binary_individual_t& individual, double const pr);

    }
}

#endif