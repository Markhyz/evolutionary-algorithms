#ifndef GUARD_H_EVO_ALG_CORE
#define GUARD_H_EVO_ALG_CORE

#include "core/fitness.hpp"
#include "core/genotype.hpp"
#include "core/individual.hpp"
#include "core/phenotype.hpp"
#include "core/population.hpp"

namespace evo_alg {
    using real_gene_t = long double;
    using integer_gene_t = int;
    using binary_gene_t = bool;

    using real_chromosome_t = std::vector<real_gene_t>;
    using integer_chromosome_t = std::vector<integer_gene_t>;
    using binary_chromosome_t = std::vector<binary_gene_t>;

    using real_individual_t = Individual<real_gene_t>;
    using integer_individual_t = Individual<integer_gene_t>;
    using binary_individual_t = Individual<binary_gene_t>;
}

#endif