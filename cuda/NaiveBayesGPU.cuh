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
    /**
    * @brief Oblicza rozmiar bloku i siatki dla podanego kernela.
    *
    * @param kernel Funkcja kernela, dla której obliczane są parametry.
    * @param dataSize Rozmiar danych, które mają zostać przetworzone.
    * @param numBlocks Liczba bloków do użycia.
    * @param blockSize Rozmiar bloku do użycia.
    * @param dynamicSharedMem Rozmiar dynamicznej pamięci współdzielonej (opcjonalnie).
    * @param strategy Strategia zaokrąglania rozmiaru bloku (domyślnie ROUND_NEAREST).
    *
    * @details
    * Funkcja ta oblicza liczbę bloków i rozmiar bloku potrzebne do efektywnego
    * uruchomienia kernela na GPU.
    *
    * @brief Calculates block and grid size for the given kernel.
    *
    * @param kernel Kernel function for which the parameters are calculated.
    * @param dataSize Size of the data to be processed.
    * @param numBlocks Number of blocks to use.
    * @param blockSize Block size to use.
    * @param dynamicSharedMem Size of the dynamic shared memory (optional).
    * @param strategy Strategy for rounding the block size (default ROUND_NEAREST).
    *
    * @details
    * This function calculates the number of blocks and block size needed to efficiently
    * launch the kernel on the GPU.
    */
    template <typename KernelFunc>
    void calculateBlockAndGridSize(KernelFunc kernel, size_t dataSize, int& numBlocks, int& blockSize,
                                   size_t dynamicSharedMem = 0, RoundingStrategy strategy = ROUND_NEAREST);

    /**
     * @brief Trenuje model Naive Bayes na podanych danych.
     *
     * @param trainLabels Etykiety dla próbek treningowych.
     * @param vocabulary Słownik, który mapuje słowa na indeksy.
     * @param featureVectorsCSR Macierz rzadkich wektorów cech w formacie CSR.
     *
     * @details
     * Metoda ta trenuje model Naive Bayes, obliczając odpowiednie prawdopodobieństwa
     * klas i cech na podstawie podanych danych treningowych.
     *
     * @brief Trains the Naive Bayes model on the provided data.
     *
     * @param trainLabels Labels for the training samples.
     * @param vocabulary Dictionary mapping words to indices.
     * @param featureVectorsCSR Sparse feature vectors matrix in CSR format.
     *
     * @details
     * This method trains the Naive Bayes model by calculating the appropriate class
     * and feature probabilities based on the provided training data.
     */
    void train(const std::vector<int>& trainLabels, const std::unordered_map<std::string, int>& vocabulary,
               const CSRMatrix& featureVectorsCSR);

    // [[nodiscard]] auto predict(const std::vector<int>& trainLabels, const CSRMatrix& featureVectorsCSR, int sampleIdx) -> int;

    /**
     * @brief Przewiduje etykiety klas dla grupy próbek na podstawie modelu Naive Bayes.
     *
     * @param trainLabels Etykiety klas dla próbek treningowych.
     * @param featureVectorsCSR Macierz rzadkich wektorów cech w formacie CSR.
     * @param numSamples Liczba próbek, dla których mają zostać przewidziane etykiety.
     * @return std::vector<int> Wektor przewidywanych etykiet dla próbek.
     *
     * @details
     * Metoda ta przewiduje klasy dla grupy próbek na podstawie wytrenowanego modelu Naive Bayes, używając GPU
     * do szybkiego obliczania logarytmicznych prawdopodobieństw klas.
     *
     * @brief Predicts class labels for a batch of samples based on the Naive Bayes model.
     *
     * @param trainLabels Class labels for the training samples.
     * @param featureVectorsCSR Sparse feature vectors matrix in CSR format.
     * @param numSamples Number of samples to predict.
     * @return std::vector<int> Vector of predicted labels for the samples.
     *
     * @details
     * This method predicts class labels for a batch of samples based on the trained Naive Bayes model,
     * using the GPU to efficiently calculate log probabilities for the classes.
     */
    auto predictBatch(const std::vector<int>& trainLabels,
                      const CSRMatrix& featureVectorsCSR,
                      size_t numSamples) -> std::vector<int>;

    /**
     * @brief Ewaluacja modelu Naive Bayes na podstawie podanych próbek i prawdziwych etykiet.
     *
     * @param featureVectorsCSR Macierz rzadkich wektorów cech w formacie CSR dla danych testowych.
     * @param trueLabels Prawdziwe etykiety dla danych testowych.
     * @param positiveClass Klasa uznawana za pozytywną podczas obliczania miar jakości modelu.
     *
     * @details
     * Metoda ta przewiduje etykiety dla podanych danych testowych i porównuje je z prawdziwymi etykietami,
     * obliczając miary jakości, takie jak dokładność, precyzja, czułość i F1-score.
     *
     * @brief Evaluates the Naive Bayes model based on the provided samples and true labels.
     *
     * @param featureVectorsCSR Sparse feature vectors matrix in CSR format for the test data.
     * @param trueLabels True labels for the test data.
     * @param positiveClass The class considered as positive when calculating model metrics.
     *
     * @details
     * This method predicts labels for the given test data and compares them with the true labels,
     * calculating performance metrics such as accuracy, precision, recall, and F1-score.
     */
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
