#include "NaiveBayesCPU.hpp"
#include "TextProcessor.hpp"
#include "Timer.hpp"

#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <unordered_map>

void NaiveBayesCPU::train(const std::vector<int>& trainLabels, const std::unordered_map<std::string, int>& vocabulary,
                          const CSRMatrix& featureVectorsCSR)
{
    m_vocabulary = vocabulary;

    countClasses(trainLabels);
    countFeatures(featureVectorsCSR, trainLabels);

    calculateClassProbabilities(trainLabels.size());
    calculateFeatureProbabilities();
}

auto NaiveBayesCPU::predict(const std::string& text) -> int
{
    auto featureVector = TextProcessor::textToSparseFeatureVector(m_vocabulary, text);
    std::unordered_map<int, double> logProbabilities = m_classProbabilitiesLog;

    for (const auto& label : m_classProbabilitiesLog | std::views::keys)
    {
        for (const auto& [feature_index, count] : featureVector)
        {
            if (m_featureProbabilitiesLog.at(label).contains(feature_index))
            {
                logProbabilities[label] += count * m_featureProbabilitiesLog.at(label).at(feature_index);
            }
            else
            {
                logProbabilities[label] += count * std::log(1.0 / static_cast<double>(m_vocabulary.size() + 1));
            }
        }
    }

    return std::ranges::max_element(logProbabilities, [](const auto& logProbA, const auto& logProbB)
    {
        return logProbA.second < logProbB.second;
    })->first;
}

void NaiveBayesCPU::accuracy(const ClassificationLabels& classificationLabels)
{
    size_t correct = 0;

    for (size_t i = 0; i < classificationLabels.predictedLabels.size(); ++i)
    {
        if (classificationLabels.predictedLabels[i] == classificationLabels.trueLabels[i])
        {
            correct++;
        }
    }

    m_accuracy = static_cast<double>(correct) / static_cast<double>(classificationLabels.predictedLabels.size());
}

void NaiveBayesCPU::precision(const ClassificationLabels& classificationLabels, const int positiveClass)
{
    size_t truePositives = 0;
    size_t falsePositives = 0;

    for (size_t i = 0; i < classificationLabels.predictedLabels.size(); ++i)
    {
        if (classificationLabels.predictedLabels[i] == positiveClass)
        {
            if (classificationLabels.trueLabels[i] == positiveClass)
            {
                truePositives++;
            }
            else
            {
                falsePositives++;
            }
        }
    }

    if (truePositives + falsePositives == 0)
    {
        m_precision = 0.0;
    }

    m_precision = static_cast<double>(truePositives) / (static_cast<double>(truePositives + falsePositives));
}

void NaiveBayesCPU::recall(const ClassificationLabels& classificationLabels, const int positiveClass)
{
    size_t truePositives = 0;
    size_t falseNegatives = 0;

    for (size_t i = 0; i < classificationLabels.predictedLabels.size(); ++i)
    {
        if (classificationLabels.trueLabels[i] == positiveClass)
        {
            if (classificationLabels.predictedLabels[i] == positiveClass)
            {
                truePositives++;
            }
            else
            {
                falseNegatives++;
            }
        }
    }

    if (truePositives + falseNegatives == 0)
    {
        m_recall = 0.0;
    }

    m_recall = static_cast<double>(truePositives) / (static_cast<double>(truePositives + falseNegatives));
}

void NaiveBayesCPU::f1Score()
{
    if (m_precision + m_recall == 0)
    {
        m_f1Score = 0.0;
    }

    m_f1Score = 2 * (m_precision * m_recall) / (m_precision + m_recall);
}

void NaiveBayesCPU::evaluate(const std::vector<std::string>& testTexts, const std::vector<int>& trueLabels,
                             const int positiveClass)
{
    ClassificationLabels classificationLabels;

    for (const auto& text : testTexts)
    {
        classificationLabels.predictedLabels.push_back(predict(text));
    }

    classificationLabels.trueLabels = trueLabels;

    accuracy(classificationLabels);
    precision(classificationLabels, positiveClass);
    recall(classificationLabels, positiveClass);
    f1Score();
}

void NaiveBayesCPU::printModel() const
{
    std::cout << "Model:\n";
    std::cout << "Class counts:\n";

    for (const auto& [label, count] : m_classCounts)
    {
        std::cout << "Class " << label << ": " << count << '\n';
    }

    std::cout << "Class probabilities:\n";

    for (const auto& [label, probability] : m_classProbabilitiesLog)
    {
        std::cout << "Class " << label << ": " << probability << '\n';
    }

    std::cout << "Vocabulary size: " << m_vocabulary.size() << '\n';
}

void NaiveBayesCPU::printEvaluationMetrics() const
{
    std::cout << "Evaluation metrics:\n";
    std::cout << "Accuracy: " << m_accuracy << '\n';
    std::cout << "Precision: " << m_precision << '\n';
    std::cout << "Recall: " << m_recall << '\n';
    std::cout << "F1 Score: " << m_f1Score << '\n';
}

void NaiveBayesCPU::countClasses(const std::vector<int>& trainLabels)
{
    for (const auto& label : trainLabels)
    {
        m_classCounts[label]++;
    }
}

void NaiveBayesCPU::countFeatures(const CSRMatrix& featureVectorsCSR, const std::vector<int>& trainLabels)
{
    for (size_t i = 0; i < trainLabels.size(); i++)
    {
        int label = trainLabels[i];

        size_t rowStart = featureVectorsCSR.rowPointers[i];
        size_t rowEnd = featureVectorsCSR.rowPointers[i + 1];

        for (size_t j = rowStart; j < rowEnd; j++)
        {
            size_t featureIndex = featureVectorsCSR.columnIndices[j];
            int count = featureVectorsCSR.values[j];

            m_featureCounts[label][static_cast<int>(featureIndex)] += count;
        }
    }
}

void NaiveBayesCPU::calculateClassProbabilities(std::vector<int>::size_type totalSamples)
{
    for (const auto& [label, count] : m_classCounts)
    {
        m_classProbabilitiesLog[label] = std::log(static_cast<double>(count) / static_cast<double>(totalSamples));
    }
}

void NaiveBayesCPU::calculateFeatureProbabilities()
{
    for (const auto& [label, featureCountMap] : m_featureCounts)
    {
        int totalFeatureCount = std::accumulate(featureCountMap.begin(), featureCountMap.end(), 0,
                                                [](int sum, const std::pair<int, int>& featurePair)
                                                {
                                                    return sum + featurePair.second;
                                                });


        for (const auto& featureIndex : m_vocabulary | std::views::values)
        {
            int count = featureCountMap.contains(featureIndex) ? featureCountMap.at(featureIndex) : 0;
            double probability = static_cast<double>(count + 1) / static_cast<double>(totalFeatureCount + m_vocabulary.
                size());

            m_featureProbabilitiesLog[label][featureIndex] = std::log(probability);
        }
    }
}
