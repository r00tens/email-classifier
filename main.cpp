#include "utils/CsvFileHandler.hpp"
#include "utils/Timer.hpp"

#include <iostream>

#include "utils/TextProcessor.hpp"

void load_training_dataset(const std::string& training_dataset_path, std::vector<std::vector<std::string>>& training_data)
{
    std::cout << "Loading training dataset...";

    try
    {
        Timer timer;
        timer.start();

        training_data = CsvFileHandler::read_data(training_dataset_path);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << std::endl;
        std::cerr << e.what() << std::endl;
    }
}

void extract_texts_and_labels(const std::vector<std::vector<std::string>>& data, std::vector<std::string>& texts, std::vector<int>& labels)
{
    std::cout << "Extracting texts and labels...";

    try
    {
        Timer timer;
        timer.start();

        TextProcessor::extract_texts_and_labels(data, texts, labels);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << std::endl;
        std::cerr << e.what() << std::endl;
    }
}

void build_vocabulary(const std::vector<std::string>& texts, std::unordered_map<std::string, int>& vocabulary)
{
    std::cout << "Building vocabulary...";

    try
    {
        Timer timer;
        timer.start();

        TextProcessor::build_vocabulary(texts, vocabulary);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << std::endl;
        std::cerr << e.what() << std::endl;
    }
}

void create_sparse_feature_vectors(const std::unordered_map<std::string, int>& vocabulary, const std::vector<std::string>& texts, std::vector<std::unordered_map<int, int>>& feature_vectors)
{
    std::cout << "Creating sparse feature vectors...";

    try
    {
        Timer timer;
        timer.start();

        feature_vectors = TextProcessor::create_sparse_feature_vectors(vocabulary, texts);

        timer.stop();

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " s]" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << " [FAIL]" << std::endl;
        std::cerr << e.what() << std::endl;
    }
}

int main(const int argc, char const* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <training_dataset> <test_dataset> <output>" << std::endl;
    }

    const std::string training_dataset_path = argv[1];

    std::vector<std::vector<std::string>> training_data;

    load_training_dataset(training_dataset_path, training_data);
    // CsvFileHandler::print_read_data(training_data);

    std::vector<std::string> train_texts;
    std::vector<int> train_labels;

    extract_texts_and_labels(training_data, train_texts, train_labels);
    // TextProcessor::print_texts_and_labels(train_texts, train_labels);

    std::unordered_map<std::string, int> vocabulary;

    build_vocabulary(train_texts, vocabulary);
    // TextProcessor::print_vocabulary(vocabulary);

    std::vector<std::unordered_map<int, int>> feature_vectors;
    create_sparse_feature_vectors(vocabulary, train_texts, feature_vectors);

    return 0;
}
