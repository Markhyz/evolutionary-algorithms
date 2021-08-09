#ifndef GUARD_H_EVO_ALG_PHENOTYPE
#define GUARD_H_EVO_ALG_PHENOTYPE

#include "../commons/macros.hpp"
#include "fitness.hpp"

#include <optional>
#include <stdexcept>

namespace evo_alg {
    template <typename... GeneTypes>
    class Phenotype {
      public:
        POINTER_ALIAS(Phenotype)

        class UndefinedFitnessException : public std::exception {
          public:
            virtual const char* what() const throw() {
                return "undefined fitness value";
            }
        };

        Phenotype();
        Phenotype(typename FitnessFunction<GeneTypes...>::const_shared_ptr const& fitness);
        Phenotype(Phenotype<GeneTypes...> const& phenotype);

        void evaluateFitness(Genotype<GeneTypes...> const& genotype);

        typename FitnessFunction<GeneTypes...>::const_shared_ptr getFitnessFunction() const;
        typename fitness::FitnessValue const& getFitnessValue() const;

        template <size_t ChromosomeIndex = 0>
        std::vector<std::pair<types::NthType<ChromosomeIndex, GeneTypes...>,
                              types::NthType<ChromosomeIndex, GeneTypes...>>> const
        getBounds() const;

        template <size_t ChromosomeIndex = 0, typename... Args>
        void setBounds(Args&&... args);

        void setFitnessValue(typename fitness::FitnessValue const fitness_value);

        Phenotype<GeneTypes...>& operator=(Phenotype<GeneTypes...> const& phenotype);

      private:
        typename FitnessFunction<GeneTypes...>::shared_ptr fitness_;
        std::optional<typename fitness::FitnessValue> fitness_value_;
    };

    template <typename... GeneTypes>
    Phenotype<GeneTypes...>::Phenotype(){};

    template <typename... GeneTypes>
    Phenotype<GeneTypes...>::Phenotype(typename FitnessFunction<GeneTypes...>::const_shared_ptr const& fitness)
        : fitness_(fitness->clone()){};

    template <typename... GeneTypes>
    Phenotype<GeneTypes...>::Phenotype(Phenotype<GeneTypes...> const& phenotype)
        : fitness_(phenotype.fitness_->clone()), fitness_value_(phenotype.fitness_value_){};

    template <typename... GeneTypes>
    void Phenotype<GeneTypes...>::evaluateFitness(Genotype<GeneTypes...> const& genotype) {
        fitness_value_ = (*fitness_)(genotype);
    }

    template <typename... GeneTypes>
    typename FitnessFunction<GeneTypes...>::const_shared_ptr Phenotype<GeneTypes...>::getFitnessFunction() const {
        return fitness_;
    }

    template <typename... GeneTypes>
    typename fitness::FitnessValue const& Phenotype<GeneTypes...>::getFitnessValue() const {
        if (!fitness_value_.has_value()) {
            throw UndefinedFitnessException();
        }

        return fitness_value_.value();
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    std::vector<
        std::pair<types::NthType<ChromosomeIndex, GeneTypes...>, types::NthType<ChromosomeIndex, GeneTypes...>>> const
    Phenotype<GeneTypes...>::getBounds() const {
        return fitness_->template getBounds<ChromosomeIndex>();
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex, typename... Args>
    void Phenotype<GeneTypes...>::setBounds(Args&&... args) {
        fitness_->template setBounds<ChromosomeIndex>(std::forward<Args>(args)...);
    }

    template <typename... GeneTypes>
    void Phenotype<GeneTypes...>::setFitnessValue(typename fitness::FitnessValue const fitness_value) {
        fitness_value_ = fitness_value;
    }

    template <typename... GeneTypes>
    Phenotype<GeneTypes...>& Phenotype<GeneTypes...>::operator=(Phenotype<GeneTypes...> const& phenotype) {
        fitness_ = typename FitnessFunction<GeneTypes...>::shared_ptr(phenotype.fitness_->clone());
        fitness_value_ = phenotype.fitness_value_;

        return *this;
    }
}

#endif