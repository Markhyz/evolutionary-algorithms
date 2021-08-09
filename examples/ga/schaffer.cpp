#include <evo_alg/algorithms/ga.hpp>
#include <evo_alg/core.hpp>
#include <evo_alg/operators.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

class SchafferFunction : public evo_alg::FitnessFunction<double> {
  public:
    SchafferFunction() : FitnessFunction<double>{{2, {-100, 100}}} {};

    size_t getDimension() const override {
        return dimension_;
    }

    SchafferFunction* clone() const override {
        return new SchafferFunction(*this);
    }

    FitnessValue operator()(evo_alg::Genotype<double> const& genotype) override {
        vector<double> chromosome = genotype.getChromosome();
        double x1 = chromosome[0], x2 = chromosome[1];
        double result = 0.5 + (pow(sin(x1 * x1 - x2 * x2), 2) - 0.5) / pow(1 + 0.001 * (x1 * x1 + x2 * x2), 2);

        return {-result};
    }

  private:
    size_t dimension_ = 1;
};

evo_alg::real_individual_t polynomialMutation(evo_alg::real_individual_t const& individual, double const pr) {
    return evo_alg::mutator::polynomial(individual, pr, 20);
}

pair<evo_alg::real_individual_t, evo_alg::real_individual_t> sbxCrossover(evo_alg::real_individual_t const& parent_1,
                                                                          evo_alg::real_individual_t const& parent_2) {
    return evo_alg::recombinator::sbx(parent_1, parent_2, 1, 0.5);
}

size_t tournamentSelection(std::vector<double> const& individuals_fit) {
    return evo_alg::selector::tournament(individuals_fit, 2);
}

int main() {
    evo_alg::FitnessFunction<double>::const_shared_ptr fit(new SchafferFunction());

    evo_alg::Population<evo_alg::Individual<double>> pop;
    evo_alg::Individual<double> best_ind;
    vector<double> best_fit, mean_fit, diversity;
    tie(best_ind, pop, best_fit, mean_fit, diversity) =
        evo_alg::ga<evo_alg::Individual<double>, evo_alg::FitnessFunction<double>>(
            2000, 20, 0, fit, evo_alg::initializator::uniformRandomInit<double>, evo_alg::selector::roulette,
            sbxCrossover, 0.95, polynomialMutation, 0.05, 1, NAN, 300);

    evo_alg::Individual<double> true_ind(fit, vector<double>(2, 0));
    true_ind.evaluateFitness();

    cout << fixed;
    cout.precision(9);
    cout << "true " << true_ind.getFitnessValue()[0] << " / found " << best_ind.getFitnessValue()[0] << endl;

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