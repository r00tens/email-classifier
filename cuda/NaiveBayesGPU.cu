#include "CudaTimer.cuh"
#include "NaiveBayesGPU.cuh"

#include "../utils/Timer.hpp"

#include <iostream>
#include <numeric>

/**
 * @details
 * Funkcja calculateBlockAndGridSize() wykorzystuje funkcję cudaOccupancyMaxPotentialBlockSize, aby obliczyć
 * optymalny rozmiar bloku i liczbę bloków dla podanego kernela. W zależności od wybranej strategii zaokrąglania,
 * rozmiar bloku jest odpowiednio modyfikowany.
 *
 * @details
 * The calculateBlockAndGridSize() function uses cudaOccupancyMaxPotentialBlockSize to calculate
 * the optimal block size and number of blocks for the given kernel. Depending on the selected
 * rounding strategy, the block size is adjusted accordingly.
 */
template <typename KernelFunc>
void NaiveBayesGPU::calculateBlockAndGridSize(KernelFunc kernel, const size_t dataSize, int& numBlocks, int& blockSize,
                                              size_t dynamicSharedMem, const RoundingStrategy strategy)
{
    constexpr int WARP_SIZE = 32;
    constexpr int ROUND_UP_OFFSET = WARP_SIZE - 1;
    constexpr int ROUND_NEAREST_OFFSET = WARP_SIZE / 2;

    int maxBlockSize = m_gpuInfo.getMaxThreadsPerBlock();
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, dynamicSharedMem, maxBlockSize);

    switch (strategy)
    {
    case ROUND_UP:
        blockSize = ((blockSize + ROUND_UP_OFFSET) / WARP_SIZE) * WARP_SIZE;
        break;
    case ROUND_NEAREST:
        blockSize = ((blockSize + ROUND_NEAREST_OFFSET) / WARP_SIZE) * WARP_SIZE;
        break;
    case ROUND_DOWN:
        blockSize = (blockSize / WARP_SIZE) * WARP_SIZE;
        break;
    }

    numBlocks = GPUInfo::calculateNumBlocks(static_cast<int>(dataSize), blockSize);

    numBlocks = std::max(numBlocks, minGridSize);
}

/**
 * @details
 * Kernel countClassesKernel() oblicza liczność każdej klasy, zliczając wystąpienia etykiet w zbiorze danych.
 * Wynik jest zapisywany w tablicy classCounts, używając operacji atomowej, aby zapewnić poprawność wyników
 * w środowisku wielowątkowym.
 *
 * @details
 * The countClassesKernel() calculates the count of each class by counting occurrences of labels in the dataset.
 * The result is stored in the classCounts array using an atomic operation to ensure correctness in a multi-threaded
 * environment.
 */
__global__ void countClassesKernel(int* classCounts, const int* labels, const size_t numSamples)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx < numSamples)
    {
        atomicAdd(&classCounts[labels[idx]], 1);
    }
}

/**
 * @details
 * Kernel countFeaturesKernel() iteruje przez rzadkie wektory cech dla każdej próbki, zliczając wystąpienia cech
 * (indeksów słów) dla każdej klasy. Dla każdego wiersza macierzy CSR (próbki) funkcja atomicAdd dodaje liczbę
 * wystąpień cech do odpowiedniego miejsca w tablicy zliczeń dla cech (featureCounts).
 *
 * @details
 * The countFeaturesKernel() iterates through sparse feature vectors for each sample, counting occurrences
 * of features (word indices) for each class. For each row of the CSR matrix (sample), the atomicAdd function
 * adds the number of feature occurrences to the appropriate position in the featureCounts array.
 */
__global__ void countFeaturesKernel(int* featureCounts, const int* trainLabels, const size_t* rowPointers,
                                    const size_t* columnIndices, const int* values, const size_t numSamples,
                                    const size_t vocabularySize)
{
    const unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx < numSamples)
    {
        const int label = trainLabels[idx];

        const size_t rowStart = rowPointers[idx];
        const size_t rowEnd = rowPointers[idx + 1];

        for (size_t j = rowStart; j < rowEnd; j++)
        {
            const size_t featureIndex = columnIndices[j];
            const int count = values[j];

            atomicAdd(&featureCounts[(label * vocabularySize) + featureIndex], count);
        }
    }
}

/**
 * @details
 * Kernel calculateClassProbabilitiesKernel() oblicza logarytmiczne prawdopodobieństwa klas na podstawie zliczeń
 * klas i liczby wszystkich próbek. Każdy wątek przetwarza jedną klasę, obliczając logarytm prawdopodobieństwa
 * dla tej klasy. Bloki organizują wątki, ale każdy wątek oblicza wartość dla jednej klasy, niezależnie od innych.
 *
 * @details
 * The calculateClassProbabilitiesKernel() calculates the log probabilities of classes based on class counts
 * and the total number of samples. Each thread processes one class, calculating the log probability for that class.
 * Blocks organize threads, but each thread computes the value for one class independently.
 */
__global__ void calculateClassProbabilitiesKernel(double* classProbabilitiesLog, const int* classCounts,
                                                  const size_t totalSamples, const int numClasses)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx < numClasses)
    {
        classProbabilitiesLog[idx] = log(static_cast<double>(classCounts[idx]) / static_cast<double>(totalSamples));
    }
}

/**
 * @details
 * Kernel calculateTotalFeatureCountKernel() oblicza całkowitą liczbę cech dla każdej klasy. Każdy wątek przetwarza
 * kombinację klasy i cechy (indeksu słowa). Wątek wykonuje operację atomową `atomicAdd`, aby dodać wystąpienia
 * cech dla danej klasy do globalnej tablicy `dTotalFeatureCounts`. Bloki organizują wątki, ale każdy wątek przetwarza
 * inną kombinację klasy i cechy.
 *
 * @details
 * The calculateTotalFeatureCountKernel() calculates the total count of features for each class. Each thread processes
 * a combination of class and feature (word index). The thread performs an atomic operation `atomicAdd` to add the
 * feature occurrences for that class to the global `dTotalFeatureCounts` array. Blocks organize threads, but each
 * thread processes a different class-feature combination.
 */
__global__ void calculateTotalFeatureCountKernel(int* dTotalFeatureCounts, const int* dFeatureCounts,
                                                 const size_t vocabularySize, const int numClasses)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int label = idx / vocabularySize;
    const unsigned int featureIndex = idx % vocabularySize;

    if (label < numClasses && featureIndex < vocabularySize)
    {
        atomicAdd(&dTotalFeatureCounts[label], dFeatureCounts[(label * vocabularySize) + featureIndex]);
    }
}

/**
 * @details
 * Kernel calculateFeatureProbabilitiesKernel() oblicza logarytmiczne prawdopodobieństwa cech dla każdej klasy,
 * uwzględniając wygładzanie Laplace'a. Każdy wątek przetwarza jedną kombinację klasy i cechy, korzystając ze wzoru:
 * P(feature | class) = (featureCount + 1) / (totalFeatureCount + vocabularySize). Wynik jest logarytmowany
 * i zapisany w tablicy dFeatureProbabilitiesLog.
 *
 * @details
 * The calculateFeatureProbabilitiesKernel() calculates the log probabilities of features for each class, applying
 * Laplace smoothing. Each thread processes one class-feature combination using the formula:
 * P(feature | class) = (featureCount + 1) / (totalFeatureCount + vocabularySize). The result is log-transformed and
 * stored in the dFeatureProbabilitiesLog array.
 */
__global__ void calculateFeatureProbabilitiesKernel(double* dFeatureProbabilitiesLog, const int* dFeatureCounts,
                                                    const int* dTotalFeatureCounts, const size_t vocabularySize,
                                                    const int numClasses)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const size_t label = idx / vocabularySize;
    const size_t featureIndex = idx % vocabularySize;

    if (label < numClasses && featureIndex < vocabularySize)
    {
        const int count = dFeatureCounts[(label * vocabularySize) + featureIndex];
        const int totalFeatureCount = dTotalFeatureCounts[label];
        const double probability = static_cast<double>(count + 1) / static_cast<double>(totalFeatureCount +
            vocabularySize);

        dFeatureProbabilitiesLog[(label * vocabularySize) + featureIndex] = log(probability);
    }
}

// __global__ void predictKernel(double* dLogProbabilities, const size_t* dRowPointers, const size_t* dColumnIndices,
//                               const int* dValues, const double* dClassProbabilitiesLog,
//                               const double* dFeatureProbabilitiesLog, const size_t vocabularySize, const int numClasses,
//                               const int sampleIdx)
// {
//     const unsigned int label = (blockIdx.x * blockDim.x) + threadIdx.x;
//
//     if (label < numClasses)
//     {
//         double logProb = dClassProbabilitiesLog[label];
//
//         const size_t rowStart = dRowPointers[sampleIdx];
//         const size_t rowEnd = dRowPointers[sampleIdx + 1];
//
//         for (size_t i = rowStart; i < rowEnd; ++i)
//         {
//             const size_t featureIndex = dColumnIndices[i];
//             const int count = dValues[i];
//
//             if (featureIndex < vocabularySize)
//             {
//                 logProb += count * dFeatureProbabilitiesLog[(label * vocabularySize) + featureIndex];
//             }
//             else
//             {
//                 logProb += count * log(1.0 / static_cast<double>(vocabularySize + 1));
//             }
//         }
//
//         dLogProbabilities[label] = logProb;
//     }
// }

/**
 * @details
 * Kernel predictKernel() przewiduje klasę dla każdej próbki, obliczając logarytmiczne prawdopodobieństwo dla każdej klasy.
 * Każdy wątek przetwarza jedną klasę dla danej próbki, iterując przez jej cechy i dodając odpowiednie logarytmiczne
 * prawdopodobieństwa cech do logarytmu prawdopodobieństwa klasy. Wynik dla każdej klasy jest zapisywany w tablicy
 * dLogProbabilities.
 *
 * @details
 * The predictKernel() predicts the class for each sample by calculating the log probability for each class.
 * Each thread processes one class for a given sample, iterating over its features and adding the corresponding
 * log feature probabilities to the class log probability. The result for each class is stored in the
 * dLogProbabilities array.
 */
__global__ void predictKernel(double* dLogProbabilities, const size_t* dRowPointers, const size_t* dColumnIndices,
                              const int* dValues, const double* dClassProbabilitiesLog,
                              const double* dFeatureProbabilitiesLog, const size_t vocabularySize, const int numClasses,
                              const size_t numSamples)
{
    const int sampleIdx = static_cast<int>(blockIdx.y);
    const unsigned int label = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (label < numClasses && sampleIdx < numSamples)
    {
        double logProb = dClassProbabilitiesLog[label];

        const size_t rowStart = dRowPointers[sampleIdx];
        const size_t rowEnd = dRowPointers[sampleIdx + 1];

        for (size_t i = rowStart; i < rowEnd; ++i)
        {
            const size_t featureIndex = dColumnIndices[i];
            const int count = dValues[i];

            if (featureIndex < vocabularySize)
            {
                logProb += count * dFeatureProbabilitiesLog[(label * vocabularySize) + featureIndex];
            }
            else
            {
                logProb += count * log(1.0 / static_cast<double>(vocabularySize + 1));
            }
        }

        dLogProbabilities[(sampleIdx * numClasses) + label] = logProb;
    }
}

/**
 * @details
 * Funkcja train() alokuje pamięć na urządzeniu, kopiuje dane z hosta na GPU, a następnie wywołuje odpowiednie kernele,
 * aby wytrenować model Naive Bayes. Po zakończeniu obliczeń wynikowe prawdopodobieństwa i zliczenia są kopiowane
 * z powrotem na hosta, gdzie są zapisywane w odpowiednich strukturach danych.
 *
 * @details
 * The train() function allocates memory on the device, copies data from the host to the GPU, and then invokes the
 * appropriate kernels to train the Naive Bayes model. After computations, the resulting probabilities and counts
 * are copied back to the host, where they are stored in the corresponding data structures.
 */
void NaiveBayesGPU::train(const std::vector<int>& trainLabels, const std::unordered_map<std::string, int>& vocabulary,
                          const CSRMatrix& featureVectorsCSR)
{
    Timer timer;
    timer.start();

    m_vocabulary = vocabulary;

    const size_t numSamples = trainLabels.size();
    const size_t vocabularySize = vocabulary.size();
    const int numClasses = *std::ranges::max_element(trainLabels) + 1;

    int* dTrainLabels;
    int* dClassCounts;
    size_t* dRowPointers;
    size_t* dColumnIndices;
    int* dValues;
    int* dFeatureCounts;
    double* dClassProbabilitiesLog;
    int* dTotalFeatureCounts;
    double* dFeatureProbabilitiesLog;

    cudaMalloc(&dTrainLabels, numSamples * sizeof(int));
    cudaMalloc(&dClassCounts, numClasses * sizeof(int));
    cudaMalloc(&dRowPointers, featureVectorsCSR.rowPointers.size() * sizeof(size_t));
    cudaMalloc(&dColumnIndices, featureVectorsCSR.columnIndices.size() * sizeof(size_t));
    cudaMalloc(&dValues, featureVectorsCSR.values.size() * sizeof(int));
    cudaMalloc(&dFeatureCounts, numClasses * vocabularySize * sizeof(int));
    cudaMalloc(&dClassProbabilitiesLog, numClasses * sizeof(double));
    cudaMalloc(&dTotalFeatureCounts, numClasses * sizeof(int));
    cudaMalloc(&dFeatureProbabilitiesLog, numClasses * vocabularySize * sizeof(double));

    cudaMemcpy(dTrainLabels, trainLabels.data(), numSamples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dRowPointers, featureVectorsCSR.rowPointers.data(),
               featureVectorsCSR.rowPointers.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dColumnIndices, featureVectorsCSR.columnIndices.data(),
               featureVectorsCSR.columnIndices.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dValues, featureVectorsCSR.values.data(), featureVectorsCSR.values.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemset(dClassCounts, 0, numClasses * sizeof(int));
    cudaMemset(dFeatureCounts, 0, numClasses * vocabularySize * sizeof(int));
    cudaMemset(dTotalFeatureCounts, 0, numClasses * sizeof(int));

    int blockSize;
    int numBlocks;

    CudaTimer cudaTimer;
    cudaTimer.start();

    calculateBlockAndGridSize(countClassesKernel, numSamples, numBlocks, blockSize);
    countClassesKernel<<<numBlocks, blockSize>>>(dClassCounts, dTrainLabels, numSamples);

    cudaDeviceSynchronize();

    calculateBlockAndGridSize(countFeaturesKernel, numSamples, numBlocks, blockSize);
    countFeaturesKernel<<<numBlocks, blockSize>>>(dFeatureCounts, dTrainLabels, dRowPointers, dColumnIndices, dValues,
                                                  numSamples, vocabularySize);
    cudaDeviceSynchronize();

    calculateBlockAndGridSize(calculateClassProbabilitiesKernel, numClasses, numBlocks, blockSize);
    calculateClassProbabilitiesKernel<<<numBlocks, blockSize>>>(dClassProbabilitiesLog, dClassCounts, numSamples,
                                                                numClasses);
    cudaDeviceSynchronize();

    calculateBlockAndGridSize(calculateTotalFeatureCountKernel, numClasses * vocabularySize, numBlocks, blockSize);
    calculateTotalFeatureCountKernel<<<numBlocks, blockSize>>>(dTotalFeatureCounts, dFeatureCounts, vocabularySize,
                                                               numClasses);

    cudaDeviceSynchronize();

    calculateBlockAndGridSize(calculateFeatureProbabilitiesKernel, numClasses * vocabularySize, numBlocks, blockSize);
    calculateFeatureProbabilitiesKernel<<<numBlocks, blockSize>>>(dFeatureProbabilitiesLog, dFeatureCounts,
                                                                  dTotalFeatureCounts, vocabularySize, numClasses);

    cudaDeviceSynchronize();

    cudaTimer.stop();

    std::vector<int> hClassCountsRaw(numClasses);
    std::vector<double> hClassProbabilitiesLog(numClasses);
    std::vector<int> hFeatureCounts(numClasses * vocabularySize);
    std::vector<int> hTotalFeatureCounts(numClasses);
    std::vector<double> hFeatureProbabilitiesLog(numClasses * vocabularySize);

    cudaMemcpy(hClassCountsRaw.data(), dClassCounts, numClasses * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hClassProbabilitiesLog.data(), dClassProbabilitiesLog, numClasses * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hFeatureCounts.data(), dFeatureCounts, numClasses * vocabularySize * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hTotalFeatureCounts.data(), dTotalFeatureCounts, numClasses * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hFeatureProbabilitiesLog.data(), dFeatureProbabilitiesLog, numClasses * vocabularySize * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < numClasses; ++i)
    {
        if (hClassCountsRaw[i] > 0)
        {
            m_classCounts[i] = hClassCountsRaw[i];
        }
    }

    for (int i = 0; i < numClasses; ++i)
    {
        m_classProbabilitiesLog[i] = hClassProbabilitiesLog[i];
    }

    for (int label = 0; label < numClasses; label++)
    {
        for (int featureIndex = 0; featureIndex < vocabularySize; featureIndex++)
        {
            const int count = hFeatureCounts[(label * vocabularySize) + featureIndex];

            if (count > 0)
            {
                m_featureCounts[label][featureIndex] = count;
            }
        }
    }

    for (int label = 0; label < numClasses; ++label)
    {
        for (int featureIndex = 0; featureIndex < vocabularySize; ++featureIndex)
        {
            m_featureProbabilitiesLog[label][featureIndex] = hFeatureProbabilitiesLog[(label * vocabularySize) +
                featureIndex];
        }
    }

    cudaFree(dTrainLabels);
    cudaFree(dClassCounts);
    cudaFree(dRowPointers);
    cudaFree(dColumnIndices);
    cudaFree(dValues);
    cudaFree(dFeatureCounts);
    cudaFree(dTotalFeatureCounts);
    cudaFree(dClassProbabilitiesLog);

    timer.stop();

    std::cout << "GPU: [DONE] [" << cudaTimer.getTimeInSeconds() << " s] [REAL TIME: " << timer.elapsed_time() <<
        " s]\n";
}

/**
 * @details
 * Funkcja predictBatch() przewiduje etykiety dla grupy próbek, wykorzystując macierz rzadkich wektorów cech (CSR)
 * oraz wcześniej wyliczone prawdopodobieństwa klas i cech. Najpierw alokowane są zasoby GPU, w tym pamięć dla
 * logarytmicznych prawdopodobieństw klas i cech, a także struktury danych CSR. Dane są kopiowane z hosta na urządzenie,
 * a następnie kernel predictKernel() jest uruchamiany w celu przewidzenia klas.
 * Wynikowe prawdopodobieństwa logarytmiczne dla każdej klasy są kopiowane z powrotem na hosta, gdzie następuje
 * wybór klasy o najwyższym prawdopodobieństwie dla każdej próbki. Na końcu zasoby GPU są zwalniane.
 *
 * @details
 * The predictBatch() function predicts labels for a batch of samples, using a sparse feature vector matrix (CSR) and
 * precomputed class and feature probabilities. First, GPU resources are allocated, including memory for log
 * probabilities of classes and features, as well as CSR data structures. The data is copied from the host to the
 * device, and the predictKernel() is launched to perform the label prediction. The resulting log probabilities
 * for each class are copied back to the host, where the class with the highest probability for each sample is selected.
 * Finally, the GPU resources are freed.
 */
auto NaiveBayesGPU::predictBatch(const std::vector<int>& trainLabels, const CSRMatrix& featureVectorsCSR,
                                 const size_t numSamples) -> std::vector<int>
{
    Timer timer;
    timer.start();

    const int numClasses = *std::ranges::max_element(trainLabels) + 1;
    const size_t vocabularySize = m_vocabulary.size();

    double* dLogProbabilities;
    size_t* dRowPointers;
    size_t* dColumnIndices;
    int* dValues;
    double* dClassProbabilitiesLog;
    double* dFeatureProbabilitiesLog;

    cudaMalloc(&dLogProbabilities, numSamples * numClasses * sizeof(double));
    cudaMalloc(&dRowPointers, featureVectorsCSR.rowPointers.size() * sizeof(size_t));
    cudaMalloc(&dColumnIndices, featureVectorsCSR.columnIndices.size() * sizeof(size_t));
    cudaMalloc(&dValues, featureVectorsCSR.values.size() * sizeof(int));
    cudaMalloc(&dClassProbabilitiesLog, numClasses * sizeof(double));
    cudaMalloc(&dFeatureProbabilitiesLog, numClasses * vocabularySize * sizeof(double));

    cudaMemcpy(dRowPointers, featureVectorsCSR.rowPointers.data(),
               featureVectorsCSR.rowPointers.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dColumnIndices, featureVectorsCSR.columnIndices.data(),
               featureVectorsCSR.columnIndices.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dValues, featureVectorsCSR.values.data(), featureVectorsCSR.values.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    std::vector hClassProbabilitiesLog(numClasses, 0.0);
    std::vector hFeatureProbabilitiesLog(numClasses * vocabularySize, 0.0);

    for (const auto& pair : m_classProbabilitiesLog)
    {
        int classIndex = pair.first;
        double probability = pair.second;

        if (classIndex >= 0 && classIndex < numClasses)
        {
            hClassProbabilitiesLog[classIndex] = probability;
        }
    }

    for (const auto& outerPair : m_featureProbabilitiesLog)
    {
        int classIndex = outerPair.first;
        const auto& innerMap = outerPair.second;

        for (const auto& innerPair : innerMap)
        {
            int wordIndex = innerPair.first;
            double probability = innerPair.second;

            int index = static_cast<int>(classIndex * vocabularySize) + wordIndex;

            hFeatureProbabilitiesLog[index] = probability;
        }
    }

    cudaMemcpy(dClassProbabilitiesLog, hClassProbabilitiesLog.data(), numClasses * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dFeatureProbabilitiesLog, hFeatureProbabilitiesLog.data(), numClasses * vocabularySize * sizeof(double),
               cudaMemcpyHostToDevice);

    int blockSize;
    dim3 numBlocks;
    int numBlocksX = static_cast<int>(numBlocks.x);

    calculateBlockAndGridSize(predictKernel, numClasses, numBlocksX, blockSize);

    numBlocks.x = numBlocksX;
    numBlocks.y = numSamples;

    CudaTimer cudaTimer;
    cudaTimer.start();

    predictKernel<<<numBlocks, blockSize>>>(dLogProbabilities, dRowPointers, dColumnIndices, dValues,
                                            dClassProbabilitiesLog, dFeatureProbabilitiesLog, vocabularySize,
                                            numClasses, numSamples);

    cudaDeviceSynchronize();

    cudaTimer.stop();

    std::vector<double> hLogProbabilities(numSamples * numClasses);
    cudaMemcpy(hLogProbabilities.data(), dLogProbabilities, numSamples * numClasses * sizeof(double),
               cudaMemcpyDeviceToHost);

    std::vector<int> predictedClasses(numSamples);
    using diffType = std::vector<double>::difference_type;

    for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
    {
        diffType start = static_cast<diffType>(sampleIdx) * numClasses;
        diffType end = static_cast<diffType>(sampleIdx + 1) * numClasses;

        int predictedClass = static_cast<int>(std::distance(hLogProbabilities.begin() + start,
                                                            std::ranges::max_element(
                                                                hLogProbabilities.begin() + start,
                                                                hLogProbabilities.begin() + end)));
        predictedClasses[sampleIdx] = predictedClass;
    }


    cudaFree(dLogProbabilities);
    cudaFree(dRowPointers);
    cudaFree(dColumnIndices);
    cudaFree(dValues);
    cudaFree(dClassProbabilitiesLog);
    cudaFree(dFeatureProbabilitiesLog);

    timer.stop();

    std::cout << "GPU: [DONE] [" << cudaTimer.getTimeInSeconds() << " s] [REAL TIME: " << timer.elapsed_time() <<
        " s]\n";

    return predictedClasses;
}

/**
 * @details
 * Funkcja evaluate() najpierw przewiduje etykiety dla próbek testowych przy użyciu metody predictBatch().
 * Następnie porównuje przewidywane etykiety z prawdziwymi etykietami, obliczając metryki ewaluacyjne, takie jak
 * dokładność, precyzja, czułość (recall) oraz F1-score. Wyniki są przechowywane w obiekcie m_evaluationMetrics.
 *
 * @details
 * The evaluate() function first predicts the labels for the test samples using the predictBatch() method.
 * It then compares the predicted labels with the true labels, calculating evaluation metrics such as
 * accuracy, precision, recall, and F1-score. The results are stored in the m_evaluationMetrics object.
 */
void NaiveBayesGPU::evaluate(const CSRMatrix& featureVectorsCSR, const std::vector<int>& trueLabels, int positiveClass)
{
    const size_t numSamples = trueLabels.size();

    ClassificationLabels classificationLabels;

    classificationLabels.predictedLabels = predictBatch(trueLabels, featureVectorsCSR, numSamples);
    classificationLabels.trueLabels = trueLabels;

    m_evaluationMetrics.accuracy(classificationLabels);
    m_evaluationMetrics.precision(classificationLabels, positiveClass);
    m_evaluationMetrics.recall(classificationLabels, positiveClass);
    m_evaluationMetrics.f1Score();
}

void NaiveBayesGPU::printEvaluationMetrics() const
{
    m_evaluationMetrics.printEvaluationMetrics();
}

auto NaiveBayesGPU::getVocabulary() const -> std::unordered_map<std::string, int>
{
    return m_vocabulary;
}

auto NaiveBayesGPU::getClassCounts() const -> std::unordered_map<int, int>
{
    return m_classCounts;
}

auto NaiveBayesGPU::getFeatureCounts() const -> std::unordered_map<int, std::unordered_map<int, int>>
{
    return m_featureCounts;
}

auto NaiveBayesGPU::getClassProbabilitiesLog() const -> std::unordered_map<int, double>
{
    return m_classProbabilitiesLog;
}

auto NaiveBayesGPU::getFeatureProbabilitiesLog() const -> std::unordered_map<int, std::unordered_map<int, double>>
{
    return m_featureProbabilitiesLog;
}

auto NaiveBayesGPU::getEvaluationMetrics() const -> EvaluationMetrics
{
    return m_evaluationMetrics;
}
