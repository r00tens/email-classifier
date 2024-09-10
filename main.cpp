#include "utils/CsvFileHandler.hpp"
#include "utils/NaiveBayesCPU.hpp"
#include "utils/TextProcessor.hpp"
#include "utils/Timer.hpp"

#include "data-structures/CSRMatrix.hpp"

#include <iostream>

#include "cuda/NaiveBayesGPU.cuh"

void loadTrainingDataset(const std::string& trainingDatasetPath, std::vector<std::vector<std::string>>& trainingData)
{
    std::cout << "Loading training dataset...";

    try
    {
        Timer timer;
        timer.start();

        trainingData = CsvFileHandler::readData(trainingDatasetPath);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << '\n';
        std::cerr << e.what() << '\n';
    }
}

void extractTextsAndLabels(const std::vector<std::vector<std::string>>& data, std::vector<std::string>& texts,
                           std::vector<int>& labels)
{
    std::cout << "Extracting texts and labels...";

    try
    {
        Timer timer;
        timer.start();

        TextProcessor::extractTextsAndLabels(data, texts, labels);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << '\n';
        std::cerr << e.what() << '\n';
    }
}

void buildVocabulary(const std::vector<std::string>& texts, std::unordered_map<std::string, int>& vocabulary)
{
    std::cout << "Building vocabulary...";

    try
    {
        Timer timer;
        timer.start();

        TextProcessor::buildVocabulary(texts, vocabulary);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << '\n';
        std::cerr << e.what() << '\n';
    }
}

void saveVocabulary(const std::unordered_map<std::string, int>& vocabulary, const std::string& outputFilename)
{
    std::cout << "Saving vocabulary...";

    try
    {
        Timer timer;
        timer.start();

        CsvFileHandler::writeData(outputFilename, vocabulary, "word", "index");

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << '\n';
        std::cerr << e.what() << '\n';
    }
}

void loadVocabulary(const std::string& filename, std::unordered_map<std::string, int>& vocabulary)
{
    std::cout << "Loading vocabulary...";

    try
    {
        Timer timer;
        timer.start();

        vocabulary = CsvFileHandler::readDataToMap<std::string, int>(filename);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << '\n';
        std::cerr << e.what() << '\n';

        std::exit(1);
    }
}

void createSparseFeatureVectors(const std::unordered_map<std::string, int>& vocabulary,
                                const std::vector<std::string>& texts,
                                std::vector<std::unordered_map<int, int>>& featureVectors, const size_t batchSize,
                                const std::string& outputFilename = "sparse-feature-vectors.csv",
                                const bool saveInBatches = false)
{
    if (!saveInBatches)
    {
        std::cout << "Creating sparse feature vectors...";

        try
        {
            Timer timer;
            timer.start();

            featureVectors = TextProcessor::createSparseFeatureVectors(vocabulary, texts);

            timer.stop();

            std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << '\n';

            return;
        }
        catch (const std::exception& e)
        {
            std::cout << " [FAIL]" << '\n';
            std::cerr << e.what() << '\n';
        }
    }

    std::cout << "Creating and saving sparse feature vectors in batches..." << std::flush;

    std::string previousMessage;

    try
    {
        Timer timer;
        timer.start();

        const size_t totalTexts = texts.size();
        const size_t totalBatches = (totalTexts + batchSize - 1) / batchSize;
        size_t globalIndex{};

        for (size_t i = 0; i < totalTexts; i += batchSize)
        {
            const size_t end = std::min(i + batchSize, totalTexts);

            std::vector batch(
                std::next(texts.begin(), static_cast<std::vector<std::string>::difference_type>(i)),
                std::next(texts.begin(), static_cast<std::vector<std::string>::difference_type>(end))
            );

            std::vector<std::unordered_map<int, int>> batchFeatureVectors =
                TextProcessor::createSparseFeatureVectors(vocabulary, batch);

            CsvFileHandler::writeSparseFeatureVectors(outputFilename, batchFeatureVectors, i == 0, globalIndex);

            globalIndex += batchFeatureVectors.size();

            std::ostringstream messageStream;
            messageStream << "\rCreating and saving sparse feature vectors in batches... Processed and saved batch "
                << ((i / batchSize) + 1) << " of " << totalBatches << std::flush;
            std::string message = messageStream.str();

            if (!previousMessage.empty())
            {
                std::cout << "\r" << std::string(previousMessage.size(), ' ') << "\r";
            }

            std::cout << message << std::flush;

            previousMessage = message;
        }

        timer.stop();

        if (!previousMessage.empty())
        {
            std::cout << "\r" << std::string(previousMessage.size(), ' ') << "\r";
        }

        std::cout << "Creating and saving sparse feature vectors in batches... [DONE] [" << std::fixed <<
            std::setprecision(4) << timer.elapsed_time() << " s]" << '\n';
    }
    catch (const std::exception& e)
    {
        if (!previousMessage.empty())
        {
            std::cout << "\r" << std::string(previousMessage.size(), ' ') << "\r";
        }

        std::cout << "Creating and saving sparse feature vectors in batches... [FAIL]" << '\n';
        std::cerr << e.what() << '\n';
    }
}

void loadSparseFeatureVectors(const std::string& filename, CSRMatrix& sparseFeatureVectors)
{
    std::cout << "Loading sparse feature vectors...";

    try
    {
        Timer timer;
        timer.start();

        sparseFeatureVectors = loadSparseFeatureVectorsToCSR(filename);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << '\n';
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << '\n';
        std::cerr << e.what() << '\n';
    }
}

void trainClassifier(NaiveBayesCPU& naiveBayesCPU, NaiveBayesGPU& naiveBayesGPU, const std::vector<int>& trainLabels,
                     const std::unordered_map<std::string, int>& vocabulary, const CSRMatrix& csrSparseFeatureVectors)
{
    std::cout << "Training classifier...\n";

    Timer timer;
    timer.start();

    naiveBayesCPU.train(trainLabels, vocabulary, csrSparseFeatureVectors);

    timer.stop();

    std::cout << "CPU: [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]\n";

    naiveBayesGPU.train(trainLabels, vocabulary, csrSparseFeatureVectors);
}

void evaluateClassifier(NaiveBayesCPU& naiveBayesCPU, NaiveBayesGPU& naiveBayesGPU, const CSRMatrix& csrMatrix, const std::vector<std::string>& testTexts,
                        const std::vector<int>& testLabels)
{
    std::cout << "Evaluating classifier...\n";

    Timer timer;
    timer.start();

    constexpr int POSITIVE_CLASS = 1;

    naiveBayesCPU.evaluate(testTexts, testLabels, POSITIVE_CLASS);

    timer.stop();

    std::cout << "CPU: [DONE] [" << timer.elapsed_time() << " s]\n";

    naiveBayesCPU.printEvaluationMetrics();

    naiveBayesGPU.evaluate(csrMatrix, testLabels, POSITIVE_CLASS);

    naiveBayesGPU.printEvaluationMetrics();
}

auto main(const int argc, char const* argv[]) -> int
{
    constexpr int MIN_ARGC = 5;

    if (argc != MIN_ARGC)
    {
        std::cerr << "Usage: " << argv[0] << " <training-dataset> <test-dataset> <input-dataset> <output>" << '\n';

        return 1;
    }

    const std::string trainingDatasetPath = argv[1];

    std::vector<std::vector<std::string>> trainingData;

    loadTrainingDataset(trainingDatasetPath, trainingData);

    std::vector<std::string> trainTexts;
    std::vector<int> trainLabels;

    extractTextsAndLabels(trainingData, trainTexts, trainLabels);

    std::unordered_map<std::string, int> vocabulary;

    buildVocabulary(trainTexts, vocabulary);
    saveVocabulary(vocabulary, "vocabulary.csv");
    // loadVocabulary("vocabulary.csv", vocabulary);

    std::vector<std::unordered_map<int, int>> sparseFeatureVectors;
    CSRMatrix csrSparseFeatureVectors;
    constexpr size_t BATCH_SIZE = 100000;

    // createSparseFeatureVectors(vocabulary, trainTexts, sparseFeatureVectors, BATCH_SIZE, "sparse-feature-vectors.csv");
    createSparseFeatureVectors(vocabulary, trainTexts, sparseFeatureVectors, BATCH_SIZE, "sparse-feature-vectors.csv",
                               true);

    // CSRMatrix csrSparseFeatureVectors = convertMapToCSR(sparseFeatureVectors);

    loadSparseFeatureVectors("sparse-feature-vectors.csv", csrSparseFeatureVectors);

    NaiveBayesCPU naiveBayesCPU;
    NaiveBayesGPU naiveBayesGPU;

    trainClassifier(naiveBayesCPU, naiveBayesGPU, trainLabels, vocabulary, csrSparseFeatureVectors);

    std::vector<std::vector<std::string>> testData;
    const std::string testDatasetPath = argv[2];

    loadTrainingDataset(testDatasetPath, testData);

    std::vector<std::string> testTexts;
    std::vector<int> testLabels;

    extractTextsAndLabels(testData, testTexts, testLabels);

    std::vector<std::unordered_map<int, int>> sparseFeatureVectorsTest;
    CSRMatrix csrSparseFeatureVectorsTest;

    createSparseFeatureVectors(vocabulary, testTexts, sparseFeatureVectorsTest, BATCH_SIZE, "sparse-feature-vectors-test.csv",
                               true);
    loadSparseFeatureVectors("sparse-feature-vectors-test.csv", csrSparseFeatureVectorsTest);

    evaluateClassifier(naiveBayesCPU, naiveBayesGPU, csrSparseFeatureVectorsTest, testTexts, testLabels);

    return 0;
}
