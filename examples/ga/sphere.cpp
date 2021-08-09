#include <evo_alg/algorithms/ga.hpp>
#include <evo_alg/core.hpp>
#include <evo_alg/operators.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

class SphereFunction : public evo_alg::FitnessFunction<double> {
  public:
    SphereFunction(size_t n) : FitnessFunction<double>{{n, {-5.12, 5.12}}} {};

    size_t getDimension() const override {
        return dimension_;
    }

    SphereFunction* clone() const override {
        return new SphereFunction(*this);
    }

    FitnessValue operator()(evo_alg::Genotype<double> const& genotype) override {
        vector<double> chromosome = genotype.getChromosome();
        double result = 0.0;
        for (double value : chromosome)
            result += value * value;
        return {-result};
    }

  private:
    size_t dimension_ = 1;
};

evo_alg::real_individual_t polynomialMutation(evo_alg::real_individual_t const& individual, double const pr) {
    return evo_alg::mutator::polynomial(individual, pr, 60);
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

    evo_alg::FitnessFunction<double>::const_shared_ptr fit(new SphereFunction(n));

    evo_alg::Population<evo_alg::Individual<double>> pop;
    evo_alg::Individual<double> best_ind;
    vector<double> best_fit, mean_fit, diversity;
    tie(best_ind, pop, best_fit, mean_fit, diversity) =
        evo_alg::ga<evo_alg::Individual<double>, evo_alg::FitnessFunction<double>>(
            2000, 50, 0, fit, evo_alg::initializator::uniformRandomInit<double>, evo_alg::selector::roulette,
            sbxCrossover, 0.95, polynomialMutation, 1 / (double) n, 1);

    evo_alg::Individual<double> true_ind(fit, vector<double>(n, 0));
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