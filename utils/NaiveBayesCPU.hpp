#ifndef MULTINOMIALNBC_HPP
#define MULTINOMIALNBC_HPP

#include <CSRMatrix.hpp>
#include <ClassificationLabels.hpp>

#include <string>
#include <unordered_map>
#include <vector>

class NaiveBayesCPU
{
public:
    void train(const std::vector<int>& trainLabels, const std::unordered_map<std::string, int>& vocabulary,
               const CSRMatrix& featureVectorsCSR);

    auto predict(const std::string& text) -> int;

    void evaluate(const std::vector<std::string>& testTexts, const std::vector<int>& trueLabels, int positiveClass);

    void printModel() const;
    void printEvaluationMetrics() const;

private:
    std::unordered_map<std::string, int> m_vocabulary;

    std::unordered_map<int, int> m_classCounts;
    std::unordered_map<int, std::unordered_map<int, int>> m_featureCounts;
    std::unordered_map<int, double> m_classProbabilitiesLog;
    std::unordered_map<int, std::unordered_map<int, double>> m_featureProbabilitiesLog;

    double m_accuracy{};
    double m_precision{};
    double m_recall{};
    double m_f1Score{};

    void countClasses(const std::vector<int>& trainLabels);
    void countFeatures(const CSRMatrix& featureVectorsCSR, const std::vector<int>& trainLabels);
    void calculateClassProbabilities(std::vector<int>::size_type totalSamples);
    void calculateFeatureProbabilities();

    void accuracy(const ClassificationLabels& classificationLabels);
    void precision(const ClassificationLabels& classificationLabels, int positiveClass);
    void recall(const ClassificationLabels& classificationLabels, int positiveClass);
    void f1Score();
};

#endif //MULTINOMIALNBC_HPP
