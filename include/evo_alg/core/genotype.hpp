#ifndef GUARD_H_EVO_ALG_GENOTYPE
#define GUARD_H_EVO_ALG_GENOTYPE

#include "../commons/macros.hpp"
#include "../commons/types.hpp"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace evo_alg {
    template <typename... GeneTypes>
    class Genotype {
      public:
        POINTER_ALIAS(Genotype<GeneTypes...>)

        using gene_t = std::variant<GeneTypes...>;
        using chromosome_t = std::vector<gene_t>;

        Genotype();
        Genotype(std::vector<GeneTypes> const&... chromosomes);

        template <size_t ChromosomeIndex = 0>
        std::vector<types::NthType<ChromosomeIndex, GeneTypes...>> getChromosome() const;

        template <size_t ChromosomeIndex = 0>
        void setChromosome(std::vector<types::NthType<ChromosomeIndex, GeneTypes...>> const& chromosome);

        template <size_t ChromosomeIndex = 0>
        void setChromosome(size_t const index, types::NthType<ChromosomeIndex, GeneTypes...> const& gene);

      private:
        std::vector<chromosome_t> chromosomes_;
    };

    template <typename... GeneTypes>
    Genotype<GeneTypes...>::Genotype()
        : chromosomes_{sizeof...(GeneTypes), typename Genotype<GeneTypes...>::chromosome_t()} {};

    template <typename... GeneTypes>
    Genotype<GeneTypes...>::Genotype(std::vector<GeneTypes> const&... chromosomes)
        : chromosomes_{typename Genotype<GeneTypes...>::chromosome_t(chromosomes.begin(), chromosomes.end())...} {};

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    std::vector<types::NthType<ChromosomeIndex, GeneTypes...>> Genotype<GeneTypes...>::getChromosome() const {
        static_assert(ChromosomeIndex < sizeof...(GeneTypes));

        std::vector<types::NthType<ChromosomeIndex, GeneTypes...>> chromosome;
        std::transform(chromosomes_[ChromosomeIndex].begin(), chromosomes_[ChromosomeIndex].end(),
                       std::back_inserter(chromosome),
                       [](gene_t const& gene) { return std::get<ChromosomeIndex>(gene); });

        return chromosome;
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    void Genotype<GeneTypes...>::setChromosome(
        std::vector<types::NthType<ChromosomeIndex, GeneTypes...>> const& chromosome) {
        static_assert(ChromosomeIndex < sizeof...(GeneTypes));

        chromosomes_[ChromosomeIndex] =
            typename Genotype<GeneTypes...>::chromosome_t(chromosome.begin(), chromosome.end());
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    void Genotype<GeneTypes...>::setChromosome(size_t const index,
                                               types::NthType<ChromosomeIndex, GeneTypes...> const& gene) {
        static_assert(ChromosomeIndex < sizeof...(GeneTypes));

        if (index >= chromosomes_[ChromosomeIndex].size()) {
            throw std::out_of_range("out of range access");
        }

        chromosomes_[ChromosomeIndex][index] = gene;
    }
}

#endif