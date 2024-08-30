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

    TextProcessor::extract_texts_and_labels(training_data, train_texts, train_labels);

    // for (unsigned int i{1}; const auto& text : train_texts)
    // {
    //     std::cout << i++ << ": " << text << std::endl;
    // }
    //
    // for (unsigned int i{1}; const auto& label : train_labels)
    // {
    //     std::cout << i++ << ": " << label << std::endl;
    // }

    return 0;
}
