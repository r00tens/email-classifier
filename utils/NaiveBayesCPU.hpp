#ifndef NAIVEBAYESCPU_HPP
#define NAIVEBAYESCPU_HPP

#include "EvaluationMetrics.hpp"
#include <CSRMatrix.hpp>

#include "../cuda/NaiveBayesGPU.cuh"

#include <string>
#include <unordered_map>
#include <vector>

class NaiveBayesCPU
{
public:
    void train(const std::vector<int>& trainLabels, const std::unordered_map<std::string, int>& vocabulary,
               const CSRMatrix& featureVectorsCSR);

    auto predict(const std::string& text) -> int;
    auto predictBatch(const std::vector<std::unordered_map<int, int>>& featureVectorsTest) -> std::vector<int>;

    void evaluate(const std::vector<std::unordered_map<int, int>>& featureVectorsTest,
                  const std::vector<int>& trueLabels, int positiveClass);

    void printModel() const;
    void printEvaluationMetrics() const;

    [[nodiscard]] auto getVocabulary() const -> std::unordered_map<std::string, int>;

    [[nodiscard]] auto getClassCounts() const -> std::unordered_map<int, int>;
    [[nodiscard]] auto getFeatureCounts() const -> std::unordered_map<int, std::unordered_map<int, int>>;
    [[nodiscard]] auto getClassProbabilitiesLog() const -> std::unordered_map<int, double>;
    [[nodiscard]] auto getFeatureProbabilitiesLog() const -> std::unordered_map<int, std::unordered_map<int, double>>;

    [[nodiscard]] auto getEvaluationMetrics() const -> EvaluationMetrics;

    auto operator==(const NaiveBayesGPU& other) const -> bool;

private:
    std::unordered_map<std::string, int> m_vocabulary;

    std::unordered_map<int, int> m_classCounts;
    std::unordered_map<int, std::unordered_map<int, int>> m_featureCounts;
    std::unordered_map<int, double> m_classProbabilitiesLog;
    std::unordered_map<int, std::unordered_map<int, double>> m_featureProbabilitiesLog;

    void countClasses(const std::vector<int>& trainLabels);
    void countFeatures(const CSRMatrix& featureVectorsCSR, const std::vector<int>& trainLabels);
    void calculateClassProbabilities(std::vector<int>::size_type totalSamples);
    void calculateFeatureProbabilities();

    EvaluationMetrics m_evaluationMetrics;
};

#endif //NAIVEBAYESCPU_HPP
