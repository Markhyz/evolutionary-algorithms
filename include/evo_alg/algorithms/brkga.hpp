#ifndef GUARD_H_EVO_ALG_GA
#define GUARD_H_EVO_ALG_GA

#include "../../../include/evo_alg/commons/utils.hpp"
#include "../../../include/evo_alg/core.hpp"
#include "../../../include/evo_alg/operators.hpp"

#include <omp.h>

#include <chrono>
#include <cstdio>
#include <iostream>

namespace evo_alg {
    namespace brkga {
        using decoding_function_t = std::function<std::vector<double>(real_chromosome_t const&)>;

        template <class FitnessType>
        class BrkgaFitness : public FitnessFunction<real_gene_t> {
          public:
            BrkgaFitness(size_t chromosome_size, typename FitnessType::const_shared_ptr fitness,
                         decoding_function_t decoder)
                : FitnessFunction<real_gene_t>({chromosome_size, {0, 1}}), chromosome_size_(chromosome_size),
                  fitness_(fitness->clone()), decoder_(decoder){};

            size_t getDimension() const override {
                return fitness_->getDimension();
            }

            vector<int8_t> getDirection() const override {
                return fitness_->getDirection();
            }

            BrkgaFitness* clone() const override {
                return new BrkgaFitness(chromosome_size_, fitness_, decoder_);
            }

            evo_alg::fitness::FitnessValue normalize(evo_alg::fitness::FitnessValue const& fitness_value) override {
                return fitness_->normalize(fitness_value);
            }

            fitness::FitnessValue operator()(evo_alg::Genotype<real_gene_t> const& genotype) override {
                real_chromosome_t encoded_chromosome = genotype.getChromosome();

                fitness::FitnessValue result = (*fitness_)({decoder_(encoded_chromosome)});

                return result;
            }

          private:
            size_t chromosome_size_;
            typename FitnessType::shared_ptr fitness_;
            decoding_function_t decoder_;
        };

        template <class IndividualType, class FitnessType>
        struct config_t {
            config_t(){};
            config_t(size_t iteration_num, size_t pop_size, size_t chromosome_size, double elite_fraction,
                     double mut_fraction, double elite_cross_pr, typename FitnessType::const_shared_ptr fitness,
                     decoding_function_t decoder)
                : iteration_num(iteration_num), pop_size(pop_size), chromosome_size(chromosome_size),
                  elite_fraction(elite_fraction), mut_fraction(mut_fraction), elite_cross_pr(elite_cross_pr),
                  fitness(fitness), decoder(decoder), archive_size(pop_size){};

            size_t iteration_num;
            size_t pop_size;
            size_t chromosome_size;
            double elite_fraction;
            double mut_fraction;
            double elite_cross_pr;
            typename FitnessType::const_shared_ptr fitness;
            decoding_function_t decoder;
            size_t archive_size;
            size_t log_step = 0;
            double diversity_threshold = 0.0;
            double diversity_enforcement = 1.0;
            double convergence_threshold = NAN;
            std::vector<real_chromosome_t> initial_pop;
            std::function<void(config_t&, size_t)> update_fn;
            evo_alg::diversity_function_t<IndividualType> diversity_fn;
        };

        template <class IndividualType, class FitnessType>
        std::tuple<Population<IndividualType>, std::vector<double>, std::vector<double>, std::vector<double>>
        run(config_t<IndividualType, FitnessType> config) {
            size_t const iteration_num = config.iteration_num;
            size_t const pop_size = config.pop_size;
            size_t const chromosome_size = config.chromosome_size;
            typename FitnessType::const_shared_ptr const fitness = config.fitness;
            decoding_function_t decoder = config.decoder;
            std::vector<real_chromosome_t> const& initial_pop = config.initial_pop;

            FitnessFunction<real_gene_t>::shared_ptr brkga_fitness(
                new BrkgaFitness<FitnessType>(chromosome_size, fitness, decoder));

            size_t elite_size = (size_t)((double) pop_size * config.elite_fraction);
            size_t mut_size = (size_t)((double) pop_size * config.mut_fraction);
            size_t cross_size = pop_size - elite_size - mut_size;

            std::vector<double> best_fit, mean_fit, diversity;

            std::chrono::duration<double, std::milli> it_time, gen_time, eval_time;
            std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2, tt1, tt2;

            Population<real_individual_t> population(pop_size), new_population(pop_size);
            Population<IndividualType> decoded_population(pop_size);

#pragma omp parallel for
            for (size_t index = 0; index < pop_size; ++index) {
                population[index] = {brkga_fitness};
                new_population[index] = {brkga_fitness};
                decoded_population[index] = {fitness};
            }
            if (pop_size - initial_pop.size() > 0) {
                initializator::uniformRandomInit(population, brkga_fitness, pop_size - initial_pop.size());
            }
            for (size_t index = 0; index < initial_pop.size(); ++index) {
                population[pop_size - initial_pop.size() + index].setChromosome(initial_pop[index]);
            }

            population.evaluateFitness();

            real_individual_t child(population[0]);

            best_fit.push_back(population.getBestFitness());
            mean_fit.push_back(population.getMeanFitness());
            diversity.push_back(population.getDiversity(0, population.getSize(), config.diversity_fn));

            for (size_t it = 0; it < iteration_num; ++it) {
                tt1 = std::chrono::high_resolution_clock::now();

                initializator::uniformRandomInit(new_population, brkga_fitness, mut_size);

                std::vector<size_t> sorted_individuals = population.getSortedIndividuals();
                std::vector<size_t> diversified_individuals, remaining_individuals;
                size_t allow_bad = (size_t)(elite_size * (1.0 - config.diversity_enforcement));
                for (size_t index = 0; index < sorted_individuals.size(); ++index) {
                    if (diversified_individuals.size() < elite_size) {
                        real_individual_t const& ind_1 = population[sorted_individuals[index]];
                        bool bad = false;
                        for (size_t index_2 = 0; index_2 < diversified_individuals.size(); ++index_2) {
                            real_individual_t const& ind_2 = population[diversified_individuals[index_2]];
                            double const distance = ind_1.getEuclidianDistance(ind_2);
                            if (distance < config.diversity_threshold) {
                                bad = true;
                            }
                        }
                        if (!bad || allow_bad) {
                            if (allow_bad) {
                                --allow_bad;
                            }
                            diversified_individuals.push_back(sorted_individuals[index]);
                            continue;
                        }
                    }
                    remaining_individuals.push_back(sorted_individuals[index]);
                }
                for (size_t index = 0; index < remaining_individuals.size(); ++index) {
                    diversified_individuals.push_back(remaining_individuals[index]);
                }

                for (size_t index = 0; index < elite_size; ++index) {
                    const size_t ind_index = diversified_individuals[index];
                    new_population[mut_size + index].setChromosome(population[ind_index].getChromosome());
                }

                t1 = std::chrono::high_resolution_clock::now();
                for (size_t index = 0; index < cross_size; ++index) {
                    std::uniform_int_distribution<size_t> elite_dist(0, elite_size - 1);
                    std::uniform_int_distribution<size_t> non_elite_dist(elite_size, pop_size - 1);

                    size_t const elite_parent = diversified_individuals[elite_dist(utils::rng)];
                    size_t const non_elite_parent = diversified_individuals[non_elite_dist(utils::rng)];

                    recombinator::eliteUniform<real_gene_t>(population[elite_parent], population[non_elite_parent],
                                                            child, config.elite_cross_pr);

                    new_population[mut_size + elite_size + index].setChromosome(child.getChromosome());
                }
                t2 = std::chrono::high_resolution_clock::now();
                gen_time = t2 - t1;

                assert(mut_size + elite_size + cross_size == pop_size);

                t1 = std::chrono::high_resolution_clock::now();
                new_population.evaluateFitness();
                t2 = std::chrono::high_resolution_clock::now();
                eval_time = t2 - t1;

                for (size_t index = 0; index < pop_size; ++index) {
                    population[index].setChromosome(new_population[index].getChromosome());
                    population[index].setFitnessValue(new_population[index].getFitnessValue());
                }

                double const best_fitness = population[population.getBestIndividuals()[0]].getFitnessValue()[0];
                best_fit.push_back(best_fitness);

                double const mean_fitness = population.getMeanFitness(mut_size);
                mean_fit.push_back(mean_fitness);

                double const diver = population.getDiversity(mut_size);
                diversity.push_back(diver);

                tt2 = std::chrono::high_resolution_clock::now();
                it_time = tt2 - tt1;

                if (config.log_step > 0 && it % config.log_step == 0) {
                    printf(
                        "Gen %lu -> best fitness: %.9f | mean fitness: %.9f | diversity: %.5f | e: %.2f | m: %.2f | c: "
                        "%.2f | dt: %.2f | de: %.2f\n",
                        it, best_fitness, mean_fitness, diver, config.elite_fraction, config.mut_fraction,
                        config.elite_cross_pr, config.diversity_threshold, config.diversity_enforcement);
                    printf("%*s it: %.0fms | gen: %.0fms | eval: %.0fms\n", 7 + (int) ceil(log10(it ? it : 1)), "",
                           it_time.count(), gen_time.count(), eval_time.count());
                }

                if (config.update_fn)
                    config.update_fn(config, it);

                elite_size = (size_t)((double) pop_size * config.elite_fraction);
                mut_size = (size_t)((double) pop_size * config.mut_fraction);
                cross_size = pop_size - elite_size - mut_size;

                if (!std::isnan(config.convergence_threshold) &&
                    utils::numericGreater(best_fitness, config.convergence_threshold))
                    break;
            }

            for (size_t index = 0; index < pop_size; ++index) {
                decoded_population[index].setChromosome(decoder(population[index].getChromosome()));
                decoded_population[index].setFitnessValue(population[index].getFitnessValue());
            }

            return {decoded_population, best_fit, mean_fit, diversity};
        }

        template <class PopulationType>
        void updateArchive(PopulationType const& pop, PopulationType& archive, size_t& archive_current_size) {
            std::vector<real_chromosome_t> total_chromosomes;
            std::vector<fitness::FitnessValue> total_fitness_values;

            for (size_t index = 0; index < pop.getSize(); ++index) {
                total_chromosomes.push_back(pop[index].getChromosome());
                total_fitness_values.push_back(pop[index].getFitnessValue());
            }

            for (size_t index = 0; index < archive_current_size; ++index) {
                total_chromosomes.push_back(archive[index].getChromosome());
                total_fitness_values.push_back(archive[index].getFitnessValue());
            }

            std::vector<std::vector<size_t>> frontiers = fitness::nonDominatedSorting(total_fitness_values, true);

            size_t archive_size = 0;
            for (std::vector<size_t>& frontier : frontiers) {
                if (archive_size == archive.getSize()) {
                    break;
                }
                for (size_t ind : frontier) {
                    if (archive_size == archive.getSize()) {
                        break;
                    }
                    archive[archive_size].setChromosome(total_chromosomes[ind]);
                    archive[archive_size].setFitnessValue(total_fitness_values[ind]);
                    ++archive_size;
                }
            }
            archive_current_size = archive_size;
        }

        template <class IndividualType, class FitnessType>
        std::tuple<Population<IndividualType>, Population<IndividualType>,
                   std::vector<std::tuple<std::vector<real_chromosome_t>, fitness::frontier_t>>, std::vector<double>>
        runMultiObjective(config_t<IndividualType, FitnessType> config) {
            utils::Timer timer;

            size_t const iteration_num = config.iteration_num;
            size_t const pop_size = config.pop_size;
            size_t const archive_size = config.archive_size;
            size_t const chromosome_size = config.chromosome_size;
            typename FitnessType::const_shared_ptr const fitness = config.fitness;
            decoding_function_t decoder = config.decoder;
            std::vector<real_chromosome_t> const& initial_pop = config.initial_pop;

            FitnessFunction<real_gene_t>::shared_ptr brkga_fitness(
                new BrkgaFitness<FitnessType>(chromosome_size, fitness, decoder));

            size_t elite_size = (size_t)((double) pop_size * config.elite_fraction);
            size_t mut_size = (size_t)((double) pop_size * config.mut_fraction);
            size_t cross_size = pop_size - elite_size - mut_size;

            Population<real_individual_t> population(pop_size), new_population(pop_size), archive(archive_size);
            Population<IndividualType> decoded_population(pop_size), decoded_archive(archive_size);

            std::vector<real_individual_t> best_individuals(pop_size);

#pragma omp parallel for
            for (size_t index = 0; index < pop_size; ++index) {
                population[index] = {brkga_fitness};
                new_population[index] = {brkga_fitness};
                best_individuals[index] = {brkga_fitness};
                decoded_population[index] = {fitness};
            }

#pragma omp parallel for
            for (size_t index = 0; index < archive_size; ++index) {
                archive[index] = {brkga_fitness};
                decoded_archive[index] = {fitness};
            }

            if (pop_size - initial_pop.size() > 0) {
                initializator::uniformRandomInit(population, brkga_fitness, pop_size - initial_pop.size());
            }
            for (size_t index = 0; index < initial_pop.size(); ++index) {
                population[pop_size - initial_pop.size() + index].setChromosome(initial_pop[index]);
            }

            population.evaluateFitness();

            real_individual_t child(population[0]);

            size_t archive_current_size = 0;

            updateArchive(population, archive, archive_current_size);

            std::vector<std::tuple<std::vector<real_chromosome_t>, fitness::frontier_t>> best_frontiers;
            std::vector<double> diversity;

            std::vector<size_t> best_individuals_index = archive.getBestIndividuals();
            fitness::frontier_t best_frontier;
            std::vector<real_chromosome_t> best_individuals_chromosome;
            for (size_t ind : best_individuals_index) {
                best_frontier.push_back(archive[ind].getFitnessValue());
                best_individuals_chromosome.push_back(archive[ind].getChromosome());
            }
            best_frontiers.emplace_back(best_individuals_chromosome, best_frontier);

            for (size_t index = 0; index < pop_size; ++index) {
                decoded_population[index].setChromosome(decoder(population[index].getChromosome()));
                decoded_population[index].setFitnessValue(population[index].getFitnessValue());
            }
            diversity.push_back(decoded_population.getDiversity());

            for (size_t it = 1; it < iteration_num; ++it) {
                timer.startTimer("it_time");

                initializator::uniformRandomInit(new_population, brkga_fitness, mut_size);

                std::vector<std::vector<size_t>> frontiers = population.getFrontiers(true);
                std::vector<size_t> sorted_individuals;
                for (std::vector<size_t> const& frontier : frontiers) {
                    for (size_t ind : frontier) {
                        sorted_individuals.push_back(ind);
                    }
                }

                std::vector<size_t> diversified_individuals, remaining_individuals;
                size_t allow_bad = (size_t)(elite_size * (1.0 - config.diversity_enforcement));
                for (size_t index = 0; index < sorted_individuals.size(); ++index) {
                    if (diversified_individuals.size() < elite_size) {
                        IndividualType const& ind_1 = decoded_population[sorted_individuals[index]];
                        bool good = true;
                        for (size_t index_2 = 0; index_2 < diversified_individuals.size(); ++index_2) {
                            IndividualType const& ind_2 = decoded_population[diversified_individuals[index_2]];
                            double const distance = ind_1.getEuclidianDistance(ind_2);
                            if (distance < config.diversity_threshold) {
                                good = false;
                                break;
                            }
                        }
                        if (good || allow_bad) {
                            if (allow_bad) {
                                --allow_bad;
                            }
                            diversified_individuals.push_back(sorted_individuals[index]);
                            continue;
                        }
                    }
                    remaining_individuals.push_back(sorted_individuals[index]);
                }
                for (size_t index = 0; index < remaining_individuals.size(); ++index) {
                    diversified_individuals.push_back(remaining_individuals[index]);
                }

                for (size_t index = 0; index < elite_size; ++index) {
                    const size_t ind_index = diversified_individuals[index];
                    new_population[mut_size + index].setChromosome(population[ind_index].getChromosome());
                }

                timer.startTimer("gen_time");
                for (size_t index = 0; index < cross_size; ++index) {
                    std::uniform_int_distribution<size_t> elite_dist(0, elite_size - 1);
                    std::uniform_int_distribution<size_t> non_elite_dist(elite_size, pop_size - 1);

                    size_t const elite_parent = diversified_individuals[elite_dist(utils::rng)];
                    size_t const non_elite_parent = diversified_individuals[non_elite_dist(utils::rng)];

                    recombinator::eliteUniform<real_gene_t>(population[elite_parent], population[non_elite_parent],
                                                            child, config.elite_cross_pr);

                    new_population[mut_size + elite_size + index].setChromosome(child.getChromosome());
                }
                timer.stopTimer("gen_time");

                assert(mut_size + elite_size + cross_size == pop_size);

                timer.startTimer("fitness_time");
                new_population.evaluateFitness();
                timer.stopTimer("fitness_time");

                for (size_t index = 0; index < pop_size; ++index) {
                    population[index].setChromosome(new_population[index].getChromosome());
                    population[index].setFitnessValue(new_population[index].getFitnessValue());
                }

                updateArchive(population, archive, archive_current_size);

                best_individuals_index = archive.getBestIndividuals();
                best_frontier.clear();
                best_individuals_chromosome.clear();
                for (size_t ind : best_individuals_index) {
                    best_frontier.push_back(archive[ind].getFitnessValue());
                    best_individuals_chromosome.push_back(archive[ind].getChromosome());
                }
                best_frontiers.emplace_back(best_individuals_chromosome, best_frontier);

                for (size_t index = 0; index < pop_size; ++index) {
                    decoded_population[index].setChromosome(decoder(population[index].getChromosome()));
                    decoded_population[index].setFitnessValue(population[index].getFitnessValue());
                }

                timer.startTimer("diver_time");
                double const diver =
                    decoded_population.getDiversity(mut_size, decoded_population.getSize(), config.diversity_fn);
                timer.stopTimer("diver_time");

                diversity.push_back(diver);

                timer.stopTimer("it_time");

                if (config.log_step > 0 && it % config.log_step == 0) {
                    printf("Gen %lu -> best frontier size: %3lu | diversity: %.3f | e: %.2f | m: "
                           "%.2f | c: %.2f | dt: %.2f | de: %.2f\n",
                           it, best_individuals_index.size(), diver, config.elite_fraction, config.mut_fraction,
                           config.elite_cross_pr, config.diversity_threshold, config.diversity_enforcement);
                    printf("|| -> it: %.0fms | gen: %.0fms | eval: %.0fms | diver: %.0fms\n", timer.getTime("it_time"),
                           timer.getTime("gen_time"), timer.getTime("fitness_time"), timer.getTime("diver_time"));
                }

                if (config.update_fn)
                    config.update_fn(config, it);

                elite_size = (size_t)((double) pop_size * config.elite_fraction);
                mut_size = (size_t)((double) pop_size * config.mut_fraction);
                cross_size = pop_size - elite_size - mut_size;
            }

            for (size_t index = 0; index < archive_size; ++index) {
                decoded_archive[index].setChromosome(decoder(archive[index].getChromosome()));
                decoded_archive[index].setFitnessValue(archive[index].getFitnessValue());
            }

            return {decoded_population, decoded_archive, best_frontiers, diversity};
        }
    }
}

#endif