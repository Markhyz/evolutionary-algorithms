#include <evo_alg/core.hpp>

#include <memory>
#include <numeric>

#include <gtest/gtest.h>

using namespace evo_alg;

class SimpleFitness : public FitnessFunction<double> {
  public:
    SimpleFitness(std::vector<std::pair<double, double>> bounds) : FitnessFunction<double>{bounds} {};

    virtual FitnessValue operator()(Genotype<double> const& genotype) override {
        std::vector<double> chromosome = genotype.getChromosome();
        double total = std::accumulate(chromosome.begin(), chromosome.end(), 0.0);

        return {total};
    }

    virtual size_t getDimension() const override {
        return dimension_;
    }

    virtual SimpleFitness* clone() const override {
        return new SimpleFitness(*this);
    }

  private:
    size_t dimension_ = 1;
};

class IndividualTest : public ::testing::Test {
  protected:
    IndividualTest() {
        bounds = {5, {0.0, 1.0}};
        f = std::make_shared<SimpleFitness>(bounds);
        chromosome = {2.7, 5.3, -15.2, 11, 57, 1.062};
        individual = Individual<double>(f, chromosome);
    }

    std::vector<std::pair<double, double>> bounds;
    std::shared_ptr<SimpleFitness> f;
    std::vector<double> chromosome;
    Individual<double> individual;
};

TEST_F(IndividualTest, Initialize) {
    std::vector<double> ind_chromosome = individual.getChromosome();
    std::vector<std::pair<double, double>> ind_bounds = individual.getBounds();

    EXPECT_EQ(chromosome.size(), ind_chromosome.size());

    for (size_t i = 0; i < chromosome.size(); ++i)
        EXPECT_DOUBLE_EQ(chromosome[i], ind_chromosome[i]);

    EXPECT_EQ(bounds.size(), ind_bounds.size());
    for (size_t i = 0; i < bounds.size(); ++i) {
        EXPECT_DOUBLE_EQ(bounds[i].first, ind_bounds[i].first);
        EXPECT_DOUBLE_EQ(bounds[i].second, ind_bounds[i].second);
    }
}

TEST_F(IndividualTest, EvaluateFitness) {
    individual.evaluateFitness();
    double total = individual.getFitnessValue()[0];

    EXPECT_DOUBLE_EQ(61.862, total);
}

class CompositeFitness : public FitnessFunction<double, bool, char> {
  public:
    CompositeFitness(std::vector<std::pair<double, double>> bounds_1, std::vector<std::pair<bool, bool>> bounds_2,
                     std::vector<std::pair<char, char>> bounds_3)
        : FitnessFunction<double, bool, char>{bounds_1, bounds_2, bounds_3} {};

    virtual FitnessValue operator()(Genotype<double, bool, char> const& genotype) override {
        std::vector<double> chromosome_1 = genotype.getChromosome<0>();
        std::vector<bool> chromosome_2 = genotype.getChromosome<1>();
        std::vector<char> chromosome_3 = genotype.getChromosome<2>();

        double total = std::accumulate(chromosome_1.begin(), chromosome_1.end(), 0.0);
        int x =
            std::accumulate(chromosome_2.begin(), chromosome_2.end(), 0, [](int tot, bool cur) { return tot + cur; });
        int y = std::accumulate(chromosome_3.begin(), chromosome_3.end(), 0,
                                [](int tot, char cur) { return tot + (cur == 'a' || cur == 't'); });

        return {total, (double) x * y};
    }

    virtual size_t getDimension() const override {
        return dimension_;
    }

    virtual CompositeFitness* clone() const override {
        return new CompositeFitness(*this);
    }

  private:
    size_t dimension_ = 1;
};

class CompositeIndividualTest : public ::testing::Test {
  protected:
    CompositeIndividualTest() {
        bounds_1 = {{-123.43, 123.43}, {0.0, 1.0}, {3001.1, 7123.134}};
        bounds_2 = {6, {0, 1}};
        bounds_3 = {8, {'a', 't'}};
        f = std::make_shared<CompositeFitness>(bounds_1, bounds_2, bounds_3);
        chromosome_1 = {5.43, -11, 103.4};
        chromosome_2 = {0, 0, 1, 1, 0, 1};
        chromosome_3 = {'a', 't', 't', 'g', 'c', 'a', 't', 'a'};
        individual = Individual<double, bool, char>(f, chromosome_1, chromosome_2, chromosome_3);
    }

    std::shared_ptr<CompositeFitness> f;
    std::vector<double> chromosome_1;
    std::vector<bool> chromosome_2;
    std::vector<char> chromosome_3;
    std::vector<std::pair<double, double>> bounds_1;
    std::vector<std::pair<bool, bool>> bounds_2;
    std::vector<std::pair<char, char>> bounds_3;
    Individual<double, bool, char> individual;
};

TEST_F(CompositeIndividualTest, Initialize) {
    std::vector<double> ind_chromosome_1 = individual.getChromosome<0>();
    std::vector<bool> ind_chromosome_2 = individual.getChromosome<1>();
    std::vector<char> ind_chromosome_3 = individual.getChromosome<2>();
    std::vector<std::pair<double, double>> ind_bounds_1 = individual.getBounds<0>();
    std::vector<std::pair<bool, bool>> ind_bounds_2 = individual.getBounds<1>();
    std::vector<std::pair<char, char>> ind_bounds_3 = individual.getBounds<2>();

    ASSERT_EQ(chromosome_1.size(), ind_chromosome_1.size());
    for (size_t i = 0; i < chromosome_1.size(); ++i)
        EXPECT_DOUBLE_EQ(chromosome_1[i], ind_chromosome_1[i]);

    ASSERT_EQ(chromosome_2.size(), ind_chromosome_2.size());
    for (size_t i = 0; i < chromosome_2.size(); ++i)
        EXPECT_EQ(chromosome_2[i], ind_chromosome_2[i]);

    ASSERT_EQ(chromosome_3.size(), ind_chromosome_3.size());
    for (size_t i = 0; i < chromosome_3.size(); ++i)
        EXPECT_EQ(chromosome_3[i], ind_chromosome_3[i]);

    EXPECT_EQ(bounds_1.size(), ind_bounds_1.size());
    for (size_t i = 0; i < bounds_1.size(); ++i) {
        EXPECT_DOUBLE_EQ(bounds_1[i].first, ind_bounds_1[i].first);
        EXPECT_DOUBLE_EQ(bounds_1[i].second, ind_bounds_1[i].second);
    }

    EXPECT_EQ(bounds_2.size(), ind_bounds_2.size());
    for (size_t i = 0; i < bounds_2.size(); ++i) {
        EXPECT_EQ(bounds_2[i].first, ind_bounds_2[i].first);
        EXPECT_EQ(bounds_2[i].second, ind_bounds_2[i].second);
    }

    EXPECT_EQ(bounds_3.size(), ind_bounds_3.size());
    for (size_t i = 0; i < bounds_3.size(); ++i) {
        EXPECT_EQ(bounds_3[i].first, ind_bounds_3[i].first);
        EXPECT_EQ(bounds_3[i].second, ind_bounds_3[i].second);
    }
}

TEST_F(CompositeIndividualTest, EvaluateFitness) {
    individual.evaluateFitness();
    std::vector<double> total = individual.getFitnessValue();

    EXPECT_DOUBLE_EQ(97.83, total[0]);
    EXPECT_EQ(18, total[1]);
}

TEST_F(CompositeIndividualTest, ChangeChromosome) {
    std::vector<double> new_chromosome_1 = {10, 20, 30};
    std::vector<bool> new_chromosome_2 = {1, 1, 1, 1, 1, 1};

    individual.setChromosome<1>(new_chromosome_2);

    std::vector<bool> ind_chromosome_2 = individual.getChromosome<1>();

    ASSERT_EQ(new_chromosome_2.size(), ind_chromosome_2.size());
    for (size_t i = 0; i < chromosome_2.size(); ++i)
        EXPECT_EQ(new_chromosome_2[i], ind_chromosome_2[i]);

    individual.setChromosome<0>(new_chromosome_1);

    std::vector<double> ind_chromosome_1 = individual.getChromosome<0>();

    ASSERT_EQ(new_chromosome_1.size(), ind_chromosome_1.size());
    for (size_t i = 0; i < chromosome_1.size(); ++i)
        EXPECT_DOUBLE_EQ(new_chromosome_1[i], ind_chromosome_1[i]);

    individual.evaluateFitness();
    std::vector<double> total = individual.getFitnessValue();

    EXPECT_DOUBLE_EQ(60, total[0]);
    EXPECT_EQ(36, total[1]);
}

TEST_F(CompositeIndividualTest, ChangeBounds) {
    std::vector<std::pair<double, double>> new_bounds_1 = {5, {0.0, 1.0}};
    std::vector<std::pair<char, char>> new_bounds_3 = {8, {'c', 'g'}};

    individual.setBounds<0>(new_bounds_1);

    std::vector<std::pair<double, double>> ind_bounds_1 = individual.getBounds<0>();

    ASSERT_EQ(new_bounds_1.size(), ind_bounds_1.size());
    for (size_t i = 0; i < bounds_1.size(); ++i) {
        EXPECT_DOUBLE_EQ(new_bounds_1[i].first, ind_bounds_1[i].first);
        EXPECT_DOUBLE_EQ(new_bounds_1[i].second, ind_bounds_1[i].second);
    }

    individual.setBounds<2>(new_bounds_3);

    std::vector<std::pair<char, char>> ind_bounds_3 = individual.getBounds<2>();

    ASSERT_EQ(new_bounds_3.size(), ind_bounds_3.size());
    for (size_t i = 0; i < bounds_3.size(); ++i) {
        EXPECT_DOUBLE_EQ(new_bounds_3[i].first, ind_bounds_3[i].first);
        EXPECT_DOUBLE_EQ(new_bounds_3[i].second, ind_bounds_3[i].second);
    }
}