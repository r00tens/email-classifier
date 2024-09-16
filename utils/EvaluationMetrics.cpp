#include "EvaluationMetrics.hpp"

#include <iomanip>
#include <iostream>

void EvaluationMetrics::accuracy(const ClassificationLabels& classificationLabels)
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

void EvaluationMetrics::precision(const ClassificationLabels& classificationLabels, const int positiveClass)
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

void EvaluationMetrics::recall(const ClassificationLabels& classificationLabels, const int positiveClass)
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

void EvaluationMetrics::f1Score()
{
    if (m_precision + m_recall == 0)
    {
        m_f1Score = 0.0;
    }

    m_f1Score = 2 * (m_precision * m_recall) / (m_precision + m_recall);
}

void EvaluationMetrics::printEvaluationMetrics() const
{
    constexpr int FIELD_WIDTH = 12;
    constexpr int TABLE_WIDTH = 61;
    constexpr int PRECISION = 4;

    std::cout << std::string(TABLE_WIDTH, '-') << '\n';

    std::cout << "| " << std::setw(FIELD_WIDTH) << "accuracy"
        << " | " << std::setw(FIELD_WIDTH) << "precision"
        << " | " << std::setw(FIELD_WIDTH) << "recall"
        << " | " << std::setw(FIELD_WIDTH) << "f1-score"
        << " | " << '\n';

    std::cout << std::string(TABLE_WIDTH, '-') << '\n';

    std::cout << "| " << std::setw(FIELD_WIDTH) << std::fixed << std::setprecision(PRECISION) << m_accuracy
        << " | " << std::setw(FIELD_WIDTH) << std::fixed << std::setprecision(PRECISION) << m_precision
        << " | " << std::setw(FIELD_WIDTH) << std::fixed << std::setprecision(PRECISION) << m_recall
        << " | " << std::setw(FIELD_WIDTH) << std::fixed << std::setprecision(PRECISION) << m_f1Score
        << " | " << '\n';

    std::cout << std::string(TABLE_WIDTH, '-') << '\n';
}

auto EvaluationMetrics::getAccuracy() const -> double
{
    return m_accuracy;
}

auto EvaluationMetrics::getPrecision() const -> double
{
    return m_precision;
}

auto EvaluationMetrics::getRecall() const -> double
{
    return m_recall;
}

auto EvaluationMetrics::getF1Score() const -> double
{
    return m_f1Score;
}

auto EvaluationMetrics::operator==(const EvaluationMetrics& other) const -> bool
{
    return m_accuracy == other.m_accuracy &&
        m_precision == other.m_precision &&
        m_recall == other.m_recall &&
        m_f1Score == other.m_f1Score;
}
