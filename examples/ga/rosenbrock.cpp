#include <evo_alg/algorithms/ga.hpp>
#include <evo_alg/core.hpp>
#include <evo_alg/operators.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

double max_value = 0;

class RosenbrockFunction : public evo_alg::FitnessFunction<double> {
  public:
    RosenbrockFunction(size_t n) : FitnessFunction<double>{{n, {-5, 10}}} {};

    size_t getDimension() const override {
        return dimension_;
    }

    RosenbrockFunction* clone() const override {
        return new RosenbrockFunction(*this);
    }

    double getTrueValue(vector<double> values) {
        double result = 0.0;
        for (size_t i = 0; i < values.size() - 1; ++i)
            result += 100 * pow(values[i + 1] - pow(values[i], 2), 2) + pow(values[i] - 1, 2);

        return result;
    }

    FitnessValue operator()(evo_alg::Genotype<double> const& genotype) override {
        vector<double> chromosome = genotype.getChromosome();
        double result = 0.0;
        for (size_t i = 0; i < chromosome.size() - 1; ++i)
            result += 100 * pow(chromosome[i + 1] - pow(chromosome[i], 2), 2) + pow(chromosome[i] - 1, 2);

        max_value = max(max_value, result);

        return {1 - result / max_value};
    }

  private:
    size_t dimension_ = 1;
};

evo_alg::real_individual_t polynomialMutation(evo_alg::real_individual_t const& individual, double const pr) {
    return evo_alg::mutator::polynomial(individual, pr, 40);
}

pair<evo_alg::real_individual_t, evo_alg::real_individual_t> sbxCrossover(evo_alg::real_individual_t const& parent_1,
                                                                          evo_alg::real_individual_t const& parent_2) {
    return evo_alg::recombinator::sbx(parent_1, parent_2, 1, 0.5);
}

size_t tournamentSelection(std::vector<double> const& individuals_fit) {
    return evo_alg::selector::tournament(individuals_fit, 2);
}

int main(int argc, char** argv) {
    size_t n = argc > 1 ? stoul(argv[1]) : 30;

    evo_alg::FitnessFunction<double>::shared_ptr fit(new RosenbrockFunction(n));

    evo_alg::Population<evo_alg::Individual<double>> pop;
    evo_alg::Individual<double> best_ind;
    vector<double> best_fit, mean_fit, diversity;
    tie(best_ind, pop, best_fit, mean_fit, diversity) =
        evo_alg::ga<evo_alg::Individual<double>, evo_alg::FitnessFunction<double>>(
            1000, 50, 0, fit, evo_alg::initializator::uniformRandomInit<double>, evo_alg::selector::roulette,
            sbxCrossover, 0.95, polynomialMutation, 1 / (double) n, 1);

    evo_alg::Individual<double> true_ind(fit, vector<double>(n, 1));
    true_ind.evaluateFitness();

    cout << fixed;
    cout.precision(9);
    cout << "true " << true_ind.getFitnessValue()[0] << " / found " << best_ind.getFitnessValue()[0] << " "
         << dynamic_pointer_cast<RosenbrockFunction>(fit)->getTrueValue(best_ind.getChromosome()) << endl;

    ofstream fit_out("res.fit"), diver_out("res.diver");
    diver_out << fixed;

    fit_out << fixed;
    fit_out.precision(9);
    for (size_t index = 0; index < best_fit.size(); ++index) {
        fit_out << index << " " << best_fit[index] << " " << mean_fit[index] << endl;
    }

    diver_out << fixed;
    diver_out.precision(9);
    for (size_t index = 0; index < diversity.size(); ++index) {
        diver_out << index << " " << diversity[index] << endl;
    }

    return 0;
}