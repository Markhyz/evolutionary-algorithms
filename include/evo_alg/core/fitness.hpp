#ifndef GUARD_H_EVO_ALG_FITNESS
#define GUARD_H_EVO_ALG_FITNESS

#include "../commons/macros.hpp"
#include "../commons/types.hpp"

#include "genotype.hpp"

#include <vector>

namespace evo_alg {
    namespace fitness {
        std::vector<double> linearScale(std::vector<double> const& fitness_values, double const c);
        std::vector<double> linearNormalization(std::vector<double> const& fitness_values, double const min_value,
                                                double const max_value);

        class FitnessValue {
          public:
            POINTER_ALIAS(FitnessValue)

            FitnessValue();
            FitnessValue(std::vector<double> values);
            FitnessValue(std::initializer_list<double> values);

            std::vector<double> const& getValues() const;

            size_t getDimension() const;

            double& operator[](size_t const index);
            double operator[](size_t const index) const;

            bool operator==(FitnessValue const& target_fit) const;
            bool operator!=(FitnessValue const& target_fit) const;
            bool operator<(FitnessValue const& target_fit) const;
            bool operator>(FitnessValue const& target_fit) const;
            bool operator<=(FitnessValue const& target_fit) const;
            bool operator>=(FitnessValue const& target_fit) const;

          private:
            std::vector<double> fitness_values_;
        };

        using frontier_t = std::vector<FitnessValue>;

        std::vector<double> crowdingDistance();
        std::vector<std::vector<size_t>> nonDominatedSorting(std::vector<FitnessValue> const& fitness_values,
                                                             bool const crowding_sort = false);
    }

    template <typename... GeneTypes>
    class FitnessFunction {
      public:
        POINTER_ALIAS(FitnessFunction<GeneTypes...>)

        using gene_bounds_t = std::variant<std::pair<GeneTypes, GeneTypes>...>;
        using chromosome_bounds_t = std::vector<gene_bounds_t>;

        FitnessFunction();
        FitnessFunction(std::vector<std::pair<GeneTypes, GeneTypes>> const&... bounds);

        virtual ~FitnessFunction() = default;

        virtual size_t getDimension() const = 0;

        template <size_t ChromosomeIndex = 0>
        std::vector<std::pair<types::NthType<ChromosomeIndex, GeneTypes...>,
                              types::NthType<ChromosomeIndex, GeneTypes...>>> const
        getBounds() const;

        void setBounds(std::vector<std::pair<GeneTypes, GeneTypes>> const&... bounds);

        template <size_t ChromosomeIndex = 0>
        void setBounds(std::vector<std::pair<types::NthType<ChromosomeIndex, GeneTypes...>,
                                             types::NthType<ChromosomeIndex, GeneTypes...>>> const& chromosome_bounds);

        template <size_t ChromosomeIndex = 0>
        void setBounds(size_t const gene_index,
                       std::pair<types::NthType<ChromosomeIndex, GeneTypes...>,
                                 types::NthType<ChromosomeIndex, GeneTypes...>> const gene_bounds);

        virtual FitnessFunction* clone() const = 0;

        virtual fitness::FitnessValue operator()(Genotype<GeneTypes...> const& genotype) = 0;

      private:
        std::vector<chromosome_bounds_t> bounds_;
    };

    template <typename... GeneTypes>
    FitnessFunction<GeneTypes...>::FitnessFunction(){};

    template <typename... GeneTypes>
    FitnessFunction<GeneTypes...>::FitnessFunction(std::vector<std::pair<GeneTypes, GeneTypes>> const&... bounds)
        : bounds_{typename FitnessFunction<GeneTypes...>::chromosome_bounds_t(bounds.begin(), bounds.end())...} {};

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    std::vector<
        std::pair<types::NthType<ChromosomeIndex, GeneTypes...>, types::NthType<ChromosomeIndex, GeneTypes...>>> const
    FitnessFunction<GeneTypes...>::getBounds() const {
        static_assert(ChromosomeIndex < sizeof...(GeneTypes));

        std::vector<
            std::pair<types::NthType<ChromosomeIndex, GeneTypes...>, types::NthType<ChromosomeIndex, GeneTypes...>>>
            bounds;
        std::transform(bounds_[ChromosomeIndex].begin(), bounds_[ChromosomeIndex].end(), std::back_inserter(bounds),
                       [](gene_bounds_t const& gene_bounds) { return std::get<ChromosomeIndex>(gene_bounds); });

        return bounds;
    }

    template <typename... GeneTypes>
    void FitnessFunction<GeneTypes...>::setBounds(std::vector<std::pair<GeneTypes, GeneTypes>> const&... bounds) {
        bounds_ = {typename FitnessFunction<GeneTypes...>::chromosome_bounds_t(bounds.begin(), bounds.end())...};
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    void FitnessFunction<GeneTypes...>::setBounds(
        std::vector<std::pair<types::NthType<ChromosomeIndex, GeneTypes...>,
                              types::NthType<ChromosomeIndex, GeneTypes...>>> const& chromosome_bounds) {
        static_assert(ChromosomeIndex < sizeof...(GeneTypes));

        bounds_[ChromosomeIndex] = chromosome_bounds_t(chromosome_bounds.begin(), chromosome_bounds.end());
    }

    template <typename... GeneTypes>
    template <size_t ChromosomeIndex>
    void FitnessFunction<GeneTypes...>::setBounds(
        size_t const gene_index,
        std::pair<types::NthType<ChromosomeIndex, GeneTypes...>, types::NthType<ChromosomeIndex, GeneTypes...>> const
            gene_bounds) {
        static_assert(ChromosomeIndex < sizeof...(GeneTypes));
        if (gene_index >= bounds_[ChromosomeIndex].size()) {
            throw std::out_of_range("out of range access");
        }

        bounds_[ChromosomeIndex][gene_index] = gene_bounds;
    }
}

#endif