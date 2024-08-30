#include "utils/CsvFileHandler.hpp"
#include "utils/Timer.hpp"

#include <iostream>

void load_training_dataset(const std::string& training_dataset_path)
{
    std::cout << "Loading training dataset...";

    try
    {
        std::vector<std::vector<std::string>> training_data = CsvFileHandler::read_data(training_dataset_path);

        std::cout << " [DONE]" << std::endl;
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

    load_training_dataset(training_dataset_path);

    return 0;
}
