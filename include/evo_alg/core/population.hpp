#ifndef GUARD_H_EVO_ALG_POPULATION
#define GUARD_H_EVO_ALG_POPULATION

#include "../commons/macros.hpp"
#include "../commons/types.hpp"
#include "individual.hpp"

#include <omp.h>

#ifdef EVO_ALG_MPI
#include "mpi.h"
#endif

#include <algorithm>
#include <iostream>

namespace evo_alg {
    template <class IndividualType>
    class Population {
      public:
        POINTER_ALIAS(Population)

        Population();
        Population(std::vector<IndividualType> const individuals);
        Population(size_t const pop_size);

        size_t getSize() const;
        IndividualType const& getIndividual(size_t const index) const;
        std::vector<size_t> getBestIndividuals() const;
        std::vector<size_t> getSortedIndividuals(size_t const fitness_index = 0) const;
        std::vector<std::vector<size_t>> getFrontiers(bool const crowding_sort = false) const;

        void setIndividual(size_t const index, IndividualType const& individual);

        void appendIndividual(IndividualType const& Individual);

        void evaluateFitness();

        void resize(size_t const new_size);

        double getBestFitness() const;
        double getMeanFitness(size_t const start = 0, size_t end = 0) const;
        double getPairwiseDiversity(size_t const start = 0, size_t end = 0) const;

        IndividualType& operator[](size_t const index);
        IndividualType const& operator[](size_t const index) const;

      protected:
        std::vector<IndividualType> population_;
    };

    template <class IndividualType>
    Population<IndividualType>::Population(){};

    template <class IndividualType>
    Population<IndividualType>::Population(std::vector<IndividualType> const individuals) : population_{individuals} {};

    template <class IndividualType>
    Population<IndividualType>::Population(size_t const pop_size) : population_{pop_size} {};

    template <class IndividualType>
    size_t Population<IndividualType>::getSize() const {
        return population_.size();
    }

    template <class IndividualType>
    IndividualType const& Population<IndividualType>::getIndividual(size_t const index) const {
        return (*this)[index];
    }

    template <class IndividualType>
    std::vector<size_t> Population<IndividualType>::getBestIndividuals() const {
        return getFrontiers()[0];
    }

    template <class IndividualType>
    std::vector<size_t> Population<IndividualType>::getSortedIndividuals(size_t const fitness_index) const {
        std::vector<size_t> sorted_individuals(population_.size());
        std::iota(sorted_individuals.begin(), sorted_individuals.end(), 0);
        std::sort(sorted_individuals.rbegin(), sorted_individuals.rend(),
                  [this, &fitness_index](size_t const ind_1, size_t const ind_2) {
                      return population_[ind_1].getFitnessValue()[fitness_index] <
                             population_[ind_2].getFitnessValue()[fitness_index];
                  });

        return sorted_individuals;
    }

    template <class IndividualType>
    std::vector<std::vector<size_t>> Population<IndividualType>::getFrontiers(bool const crowding_sort) const {
        std::vector<fitness::FitnessValue> pop_fitness(population_.size());
        for (size_t index = 0; index < population_.size(); ++index) {
            pop_fitness[index] = population_[index].getFitnessValue();
        }

        return fitness::nonDominatedSorting(pop_fitness, crowding_sort);
    }

    template <class IndividualType>
    void Population<IndividualType>::setIndividual(size_t const index, IndividualType const& individual) {
        (*this)[index] = individual;
    }

    template <class IndividualType>
    IndividualType& Population<IndividualType>::operator[](size_t const index) {
        if (index >= population_.size()) {
            throw std::out_of_range("out of range access");
        }

        return population_[index];
    }

    template <class IndividualType>
    IndividualType const& Population<IndividualType>::operator[](size_t const index) const {
        if (index >= population_.size()) {
            throw std::out_of_range("out of range access");
        }

        return population_[index];
    }

    template <class IndividualType>
    void Population<IndividualType>::appendIndividual(IndividualType const& individual) {
        population_.push_back(individual);
    }

    template <class IndividualType>
    void Population<IndividualType>::resize(size_t const new_size) {
        population_.resize(new_size);
    }

    template <class IndividualType>
    double Population<IndividualType>::getBestFitness() const {
        double best_fit = this->population_[0].getFitnessValue()[0];
        for (size_t index = 1; index < this->population_.size(); ++index)
            best_fit = std::max(best_fit, this->population_[index].getFitnessValue()[0]);

        return best_fit;
    }

    template <class IndividualType>
    double Population<IndividualType>::getMeanFitness(size_t const start, size_t end) const {
        if (end == 0)
            end = population_.size();
        double total_fit = 0;
        for (size_t index = start; index < end; ++index)
            total_fit += this->population_[index].getFitnessValue()[0];

        return total_fit / (double) (end - start);
    }

    template <class IndividualType>
    double Population<IndividualType>::getPairwiseDiversity(size_t const start, size_t end) const {
        using gene_type_t = typename std::tuple_element<0, typename IndividualType::gene_types>::type;

        if (end == 0)
            end = population_.size();

        double diversity = 0;

        std::vector<std::pair<gene_type_t, gene_type_t>> bounds = population_[0].getBounds();
        double normalization_factor = 0;
        for (size_t index = 0; index < bounds.size(); ++index)
            normalization_factor += pow(bounds[index].second - bounds[index].first, 2);
        normalization_factor = sqrt(normalization_factor);

        std::vector<double> ind_dist(population_.size());

#pragma omp parallel for
        for (size_t index = start; index < end; ++index) {
            for (size_t index2 = index + 1; index2 < end; ++index2) {
                std::vector<gene_type_t> chromosome_1 = population_[index].getChromosome();
                std::vector<gene_type_t> chromosome_2 = population_[index2].getChromosome();
                double pair_distance = 0;
                for (size_t gene_index = 0; gene_index < bounds.size(); ++gene_index) {
                    pair_distance += pow(chromosome_1[gene_index] - chromosome_2[gene_index], 2);
                }
                ind_dist[index] += sqrt(pair_distance);
            }
        }
        diversity += std::accumulate(ind_dist.begin(), ind_dist.end(), 0.0);
        diversity = (2 * diversity) / (double) ((end - start) * (end - start - 1));
        diversity /= normalization_factor;

        return diversity;
    }

#ifdef EVO_ALG_MPI
    template <class IndividualType>
    void Population<IndividualType>::evaluateFitness() {
        using gene_type_t = typename std::tuple_element<0, typename IndividualType::gene_types>::type;

        size_t const work_size = (size_t) utils::mpi_size;
        bool constexpr valid_type = std::is_same_v<gene_type_t, int> || std::is_same_v<gene_type_t, bool> ||
                                    std::is_same_v<gene_type_t, char> || std::is_same_v<gene_type_t, double>;
        if (work_size > 0 && valid_type) {
            size_t const fitness_dimension = population_[0].getFitnessFunction()->getDimension();
            double* fitness_value = (double*) malloc(fitness_dimension * sizeof(double));
            size_t const work_slice = (size_t) ceil((double) population_.size() / (double) work_size);
            for (size_t node_index = 1; node_index < work_size; ++node_index) {
                size_t const offset = work_slice * node_index;
                size_t const slice_size = std::min(work_slice, population_.size() - offset);
                MPI_Send(&slice_size, 1, MPI_INT, (int) node_index, 0, MPI_COMM_WORLD);

                for (size_t ind_index = offset; ind_index < offset + slice_size; ++ind_index) {
                    std::vector<gene_type_t> chromosome = population_[ind_index].getChromosome();
                    if constexpr (std::is_same_v<gene_type_t, int>) {
                        MPI_Send(chromosome.data(), (int) chromosome.size(), MPI_INT, (int) node_index, 0,
                                 MPI_COMM_WORLD);
                    } else if constexpr (std::is_same_v<gene_type_t, bool>) {
                        std::vector<char> bool_to_char(chromosome.begin(), chromosome.end());
                        MPI_Send(bool_to_char.data(), (int) bool_to_char.size(), MPI_BYTE, (int) node_index, 0,
                                 MPI_COMM_WORLD);
                    } else if constexpr (std::is_same_v<gene_type_t, char>) {
                        MPI_Send(chromosome.data(), (int) chromosome.size(), MPI_BYTE, (int) node_index, 0,
                                 MPI_COMM_WORLD);
                    } else if constexpr (std::is_same_v<gene_type_t, double>) {
                        MPI_Send(chromosome.data(), (int) chromosome.size(), MPI_DOUBLE, (int) node_index, 0,
                                 MPI_COMM_WORLD);
                    }
                }
            }
            for (size_t ind_index = 0; ind_index < work_slice; ++ind_index)
                population_[ind_index].evaluateFitness();

            for (size_t node_index = 1; node_index < work_size; ++node_index) {
                size_t const offset = work_slice * node_index;
                size_t slice_size = std::min(work_slice, population_.size() - offset);

                for (size_t ind_index = offset; ind_index < offset + slice_size; ++ind_index) {
                    MPI_Recv(fitness_value, (int) fitness_dimension, MPI_DOUBLE, (int) node_index, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    population_[ind_index].setFitnessValue({fitness_value, fitness_value + fitness_dimension});
                }
            }
        } else {
            for (size_t index = 0; index < population_.size(); ++index) {
                population_[index].evaluateFitness();
            }
        }
    }
#else
    template <class IndividualType>
    void Population<IndividualType>::evaluateFitness() {
#pragma omp parallel for
        for (size_t index = 0; index < population_.size(); ++index) {
            population_[index].evaluateFitness();
        }
    }
#endif
}

#endif