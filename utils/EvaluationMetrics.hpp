#ifndef EVALUATIONMETRICS_HPP
#define EVALUATIONMETRICS_HPP

#include <ClassificationLabels.hpp>

class EvaluationMetrics
{
public:
    void accuracy(const ClassificationLabels& classificationLabels);
    void precision(const ClassificationLabels& classificationLabels, int positiveClass);
    void recall(const ClassificationLabels& classificationLabels, int positiveClass);
    void f1Score();
    void printEvaluationMetrics() const;

    auto getAccuracy() const -> double;
    auto getPrecision() const -> double;
    auto getRecall() const -> double;
    auto getF1Score() const -> double;

    auto operator==(const EvaluationMetrics& other) const -> bool;

private:
    double m_accuracy{};
    double m_precision{};
    double m_recall{};
    double m_f1Score{};
};

#endif //EVALUATIONMETRICS_HPP
