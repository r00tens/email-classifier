#ifndef NAIVEBAYESGPU_CUH
#define NAIVEBAYESGPU_CUH

#include "GPUInfo.cuh"

#include "../utils/EvaluationMetrics.hpp"
#include "CSRMatrix.hpp"

#include <string>
#include <unordered_map>
#include <vector>

enum RoundingStrategy : std::uint8_t
{
    ROUND_UP,
    ROUND_NEAREST,
    ROUND_DOWN
};

class NaiveBayesGPU
{
public:
    template <typename KernelFunc>
    void calculateBlockAndGridSize(KernelFunc kernel, size_t dataSize, int& numBlocks, int& blockSize,
                                   size_t dynamicSharedMem = 0, RoundingStrategy strategy = ROUND_NEAREST);

    void train(const std::vector<int>& trainLabels, const std::unordered_map<std::string, int>& vocabulary,
               const CSRMatrix& featureVectorsCSR);

    // [[nodiscard]] auto predict(const std::vector<int>& trainLabels, const CSRMatrix& featureVectorsCSR, int sampleIdx) -> int;
    auto predictBatch(const std::vector<int>& trainLabels,
                      const CSRMatrix& featureVectorsCSR,
                      size_t numSamples) -> std::vector<int>;

    void evaluate(const CSRMatrix& featureVectorsCSR, const std::vector<int>& trueLabels, int positiveClass);

    void printEvaluationMetrics() const;

    [[nodiscard]] auto getVocabulary() const -> std::unordered_map<std::string, int>;

    [[nodiscard]] auto getClassCounts() const -> std::unordered_map<int, int>;
    [[nodiscard]] auto getFeatureCounts() const -> std::unordered_map<int, std::unordered_map<int, int>>;
    [[nodiscard]] auto getClassProbabilitiesLog() const -> std::unordered_map<int, double>;
    [[nodiscard]] auto getFeatureProbabilitiesLog() const -> std::unordered_map<int, std::unordered_map<int, double>>;

    [[nodiscard]] auto getEvaluationMetrics() const -> EvaluationMetrics;

private:
    GPUInfo m_gpuInfo;

    std::unordered_map<std::string, int> m_vocabulary;

    std::unordered_map<int, int> m_classCounts;
    std::unordered_map<int, std::unordered_map<int, int>> m_featureCounts;
    std::unordered_map<int, double> m_classProbabilitiesLog;
    std::unordered_map<int, std::unordered_map<int, double>> m_featureProbabilitiesLog;

    EvaluationMetrics m_evaluationMetrics;
};

#endif //NAIVEBAYESGPU_CUH
