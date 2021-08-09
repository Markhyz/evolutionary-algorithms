#include <evo_alg/algorithms/ga.hpp>
#include <evo_alg/core.hpp>
#include <evo_alg/operators.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

class GeneralizedPenalizedFunction : public evo_alg::FitnessFunction<double> {
  public:
    GeneralizedPenalizedFunction(size_t n) : FitnessFunction<double>{{n, {-50, 50}}} {};

    size_t getDimension() const override {
        return dimension_;
    }

    GeneralizedPenalizedFunction* clone() const override {
        return new GeneralizedPenalizedFunction(*this);
    }

    FitnessValue operator()(evo_alg::Genotype<double> const& genotype) override {
        vector<double> chromosome = genotype.getChromosome();
        double const pi = acos(-1);
        double k = 100, a = 10, m = 4;
        double t = 0, u = 0;
        for (double value : chromosome) {
            if (value > a) {
                u += k * pow(value - a, m);
            } else if (value < -a) {
                u += k * pow(-value - a, m);
            } else {
                u += 0;
            }
        }
        for (size_t i = 0; i < chromosome.size() - 1; ++i) {
            double y1 = 1 + (chromosome[i] + 1) / 4.0;
            double y2 = 1 + (chromosome[i + 1] + 1) / 4.0;
            t += pow(y1 - 1, 2) * (1 + 10 * pow(sin(pi * y2), 2));
        }
        double result = (pi * (10 * pow(sin(pi * (1 + (chromosome[0] + 1) / 4.0)), 2) + t +
                               pow((1 + (chromosome[chromosome.size() - 1] + 1) / 4.0) - 1, 2))) /
                            (double) chromosome.size() +
                        u;

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

    evo_alg::FitnessFunction<double>::const_shared_ptr fit(new GeneralizedPenalizedFunction(n));

    evo_alg::Population<evo_alg::Individual<double>> pop;
    evo_alg::Individual<double> best_ind;
    vector<double> best_fit, mean_fit, diversity;
    tie(best_ind, pop, best_fit, mean_fit, diversity) =
        evo_alg::ga<evo_alg::Individual<double>, evo_alg::FitnessFunction<double>>(
            2000, 100, 1, fit, evo_alg::initializator::uniformRandomInit<double>, tournamentSelection, sbxCrossover,
            0.95, polynomialMutation, 1 / (double) n, 1);

    evo_alg::Individual<double> true_ind(fit, vector<double>(n, -1));
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