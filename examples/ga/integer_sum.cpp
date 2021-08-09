#include <evo_alg/algorithms/ga.hpp>
#include <evo_alg/core.hpp>
#include <evo_alg/operators.hpp>

#include <fstream>
#include <iostream>

using namespace std;

class IntegerSum : public evo_alg::FitnessFunction<bool> {
  public:
    IntegerSum(vector<int> values) : FitnessFunction<bool>{{values.size(), {0, 1}}}, values_{values} {};

    size_t getDimension() const override {
        return dimension_;
    }

    IntegerSum* clone() const override {
        return new IntegerSum(*this);
    }

    FitnessValue operator()(evo_alg::Genotype<bool> const& genotype) override {
        vector<bool> chromosome = genotype.getChromosome();
        double result = 0.0;
        for (size_t i = 0; i < values_.size(); ++i)
            if (chromosome[i])
                result += values_[i];

        return {result};
    }

  private:
    vector<int> values_;
    size_t dimension_ = 1;
};

size_t tournamentSelection(std::vector<double> const& individuals_fit) {
    return evo_alg::selector::tournament(individuals_fit, 2);
}

int main(int argc, char** argv) {
    size_t n = argc > 1 ? stoul(argv[1]) : 100;

    vector<int> v(n);
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> uid(-1000, 1000);

    int true_res = 0;
    for (size_t i = 0; i < n; ++i) {
        v[i] = uid(mt);
        true_res += v[i] > 0 ? v[i] : 0;
    }

    evo_alg::FitnessFunction<bool>::const_shared_ptr fit(new IntegerSum(v));

    evo_alg::Population<evo_alg::Individual<bool>> pop;
    evo_alg::Individual<bool> best_ind;
    vector<double> best_fit, mean_fit, diversity;
    tie(best_ind, pop, best_fit, mean_fit, diversity) =
        evo_alg::ga<evo_alg::Individual<bool>, evo_alg::FitnessFunction<bool>>(
            1000, 50, 1, fit, evo_alg::initializator::uniformRandomInit<bool>, tournamentSelection,
            evo_alg::recombinator::onePoint<bool>, 0.95, evo_alg::mutator::bitFlip, 1 / (double) n, 1);

    cout << "true " << true_res << " / found " << best_ind.getFitnessValue()[0] << endl;

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