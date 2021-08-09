#ifndef GUARD_H_EVO_ALG_INDIVIDUAL
#define GUARD_H_EVO_ALG_INDIVIDUAL

#include "../commons/macros.hpp"
#include "../commons/utils.hpp"
#include "genotype.hpp"
#include "phenotype.hpp"

#include <utility>

namespace evo_alg {
    template <typename... GeneTypes>
    class Individual {
      public:
        POINTER_ALIAS(Individual)

        using gene_types = std::tuple<GeneTypes...>;

        Individual();
        Individual(typename FitnessFunction<GeneTypes...>::const_shared_ptr fitness);
        Individual(typename FitnessFunction<GeneTypes...>::const_shared_ptr fitness,
                   std::vector<GeneTypes> const&... chromosomes);

        void evaluateFitness();

        template <size_t ChromosomeIndex = 0>
        std::vector<types::NthType<ChromosomeIndex, GeneTypes...>> getChromosome() const;

        typename FitnessFunction<GeneTypes...>::const_shared_ptr getFitnessFunction() const;
        typename fitness::FitnessValue const& getFitnessValue() const;

        template <size_t ChromosomeIndex = 0>
        double getEuclidianDistance(Individual<GeneTypes...> const& target_ind) const;

        template <size_t ChromosomeIndex = 0>
        std::vector<std::pair<types::NthType<ChromosomeIndex, GeneTypes...>,
                              types::NthType<ChromosomeIndex, GeneTypes...>>> const
        getBounds() const;

        template <size_t ChromosomeIndex = 0, typename... Args>
        void setBounds(Args&&... args);

        template <size_t ChromosomeIndex = 0, typename... Args>
        void setChromosome(Args&&... args);

        void setFitnessValue(typename fitness::FitnessValue const fitness_value);

        bool operator<(Individual<GeneTypes...> const& ind) const;
        bool operator>(Individual<GeneTypes...> const& ind) const;

      private:
        Genotype<GeneTypes...> genotype_;
        Phenotype<GeneTypes...> phenotype_;
    };

    template <typename... GeneTypes>
    Individual<GeneTypes...>::Individual(){};

    template <typename... GeneTypes>
    Individual<GeneTypes...>::Individual(typename FitnessFunction<GeneTypes...>::const_shared_ptr fitness)
        : genotype_{}, phenotype_(fitness){};

    template <typename... GeneTypes>
    Individual<GeneTypes...>::Individual(typename FitnessFunction<GeneTypes...>::const_shared_ptr fitness,
                                         std::vector<GeneTypes> const&... chromosomes)
        : genotype_(chromosomes...), phenotype_(fitness){};

    template <typename... GeneTypes>
    void Individual<GeneTypes...>::evaluateFitness() {
        phenotype_.evaluateFitness(genotype_);
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    std::vector<types::NthType<ChromosomeIndex, GeneTypes...>> Individual<GeneTypes...>::getChromosome() const {
        return genotype_.template getChromosome<ChromosomeIndex>();
    }

    template <typename... GeneTypes>
    typename FitnessFunction<GeneTypes...>::const_shared_ptr Individual<GeneTypes...>::getFitnessFunction() const {
        return phenotype_.getFitnessFunction();
    }

    template <typename... GeneTypes>
    typename fitness::FitnessValue const& Individual<GeneTypes...>::getFitnessValue() const {
        return phenotype_.getFitnessValue();
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    double Individual<GeneTypes...>::getEuclidianDistance(Individual<GeneTypes...> const& target_ind) const {
        using gene_t = types::NthType<ChromosomeIndex, GeneTypes...>;

        std::vector<std::pair<gene_t, gene_t>> bounds = phenotype_.template getBounds<ChromosomeIndex>();
        double normalization_factor = 0;
        for (size_t index = 0; index < bounds.size(); ++index)
            normalization_factor += pow(bounds[index].second - bounds[index].first, 2);
        normalization_factor = sqrt(normalization_factor);

        std::vector<gene_t> chromosome = genotype_.template getChromosome<ChromosomeIndex>();
        std::vector<gene_t> target_chromosome = target_ind.template getChromosome<ChromosomeIndex>();
        double distance = 0;
        for (size_t gene_index = 0; gene_index < bounds.size(); ++gene_index) {
            distance += pow(chromosome[gene_index] - target_chromosome[gene_index], 2);
        }
        distance = sqrt(distance) / normalization_factor;

        return distance;
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    std::vector<
        std::pair<types::NthType<ChromosomeIndex, GeneTypes...>, types::NthType<ChromosomeIndex, GeneTypes...>>> const
    Individual<GeneTypes...>::getBounds() const {
        return phenotype_.template getBounds<ChromosomeIndex>();
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex, typename... Args>
    void Individual<GeneTypes...>::setBounds(Args&&... args) {
        phenotype_.template setBounds<ChromosomeIndex>(std::forward<Args>(args)...);
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex, typename... Args>
    void Individual<GeneTypes...>::setChromosome(Args&&... args) {
        genotype_.template setChromosome<ChromosomeIndex>(std::forward<Args>(args)...);
    }

    template <typename... GeneTypes>
    void Individual<GeneTypes...>::setFitnessValue(typename fitness::FitnessValue const fitness_value) {
        phenotype_.setFitnessValue(fitness_value);
    }

    template <typename... GeneTypes>
    bool Individual<GeneTypes...>::operator<(Individual<GeneTypes...> const& ind) const {
        return phenotype_.getFitnessValue() < ind.getFitnessValue();
    }

    template <typename... GeneTypes>
    bool Individual<GeneTypes...>::operator>(Individual<GeneTypes...> const& ind) const {
        return ind < *this;
    }
}

#endif