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

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " ms]" << std::endl;
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

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " ms]" << std::endl;
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

        std::cout << " [DONE] [" << std::fixed << std::setprecision(4) << timer.elapsed_time() << " ms]" << std::endl;
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

    // for (unsigned int i{1}; const auto& row : training_data)
    // {
    //     if (i == 1)
    //     {
    //         i++;
    //
    //         continue;
    //     }
    //
    //     std::cout << (i++ - 1) << ": ";
    //
    //     for (const auto& cell : row)
    //     {
    //         std::cout << cell << " ";
    //     }
    //
    //     std::cout << std::endl;
    // }

    std::vector<std::string> train_texts;
    std::vector<int> train_labels;

    extract_texts_and_labels(training_data, train_texts, train_labels);

    // for (unsigned int i{1}; const auto& text : train_texts)
    // {
    //     std::cout << i++ << ": " << text << std::endl;
    // }
    //
    // for (unsigned int i{1}; const auto& label : train_labels)
    // {
    //     std::cout << i++ << ": " << label << std::endl;
    // }

    std::unordered_map<std::string, int> vocabulary;

    build_vocabulary(train_texts, vocabulary);

    // for (unsigned int i{1}; const auto& [word, index] : vocabulary)
    // {
    //     std::cout << i++ << ": " << word << " -> " << index << std::endl;
    // }

    return 0;
}
