#ifndef CSVFILEHANDLER_HPP
#define CSVFILEHANDLER_HPP

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class CsvFileHandler
{
public:
    static auto readData(const std::string& filename) -> std::vector<std::vector<std::string>>;

    template <typename K, typename V>
    static auto readDataToMap(const std::string& filename) -> std::unordered_map<K, V>;

    template <typename K, typename V>
    static void writeData(const std::string& filename, const std::unordered_map<K, V>& data,
                          const std::string& keyColumnName = "key", const std::string& valueColumnName = "value");

    template <typename K, typename V>
    static void writeSparseFeatureVectors(const std::string& filename,
                                          const std::vector<std::unordered_map<K, V>>& sparseFeatureVectors,
                                          bool overwrite = true, size_t globalIndex = 0);

private:
    static auto parseCsvLine(const std::string& line) -> std::vector<std::string>;
    static void validateAndAddRow(std::vector<std::vector<std::string>>& data, const std::vector<std::string>& row,
                                  const std::string& line);
};

template <typename K, typename V>
auto CsvFileHandler::readDataToMap(const std::string& filename) -> std::unordered_map<K, V>
{
    std::unordered_map<K, V> data;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    bool isFirstLine = true;

    while (std::getline(file, line))
    {
        if (isFirstLine)
        {
            isFirstLine = false;
            continue;
        }

        std::vector<std::string> row = parseCsvLine(line);

        if (row.size() != 2)
        {
            throw std::runtime_error("Incorrect number of columns in row: " + line);
        }

        std::istringstream keyStream(row[0]);
        std::istringstream valueStream(row[1]);
        K key;
        V value;

        keyStream >> key;
        valueStream >> value;

        data[key] = value;
    }

    file.close();

    return data;
}

template <typename K, typename T>
void CsvFileHandler::writeData(const std::string& filename, const std::unordered_map<K, T>& data,
                               const std::string& keyColumnName, const std::string& valueColumnName)
{
    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    try
    {
        outFile << keyColumnName << "," << valueColumnName << "\n";

        for (const auto& pair : data)
        {
            outFile << pair.first << "," << pair.second << "\n";
        }

        outFile.close();

        if (outFile.fail())
        {
            throw std::runtime_error("Failed to write or close the file: " + filename);
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("An error occurred while writing to the file: " + std::string(e.what()));
    }
}

template <typename K, typename V>
void CsvFileHandler::writeSparseFeatureVectors(const std::string& filename,
                                               const std::vector<std::unordered_map<K, V>>& sparseFeatureVectors,
                                               const bool overwrite, size_t globalIndex)
{
    std::ofstream outFile;

    std::ios_base::openmode mode = std::ios_base::out;

    if (overwrite)
    {
        mode |= std::ios_base::trunc;
    }
    else
    {
        mode |= std::ios_base::app;
    }

    outFile.open(filename, mode);

    if (!outFile.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    try
    {
        if (overwrite)
        {
            outFile << "vectorIndex,key,value\n";
        }

        for (size_t i = 0; i < sparseFeatureVectors.size(); ++i)
        {
            const auto& featureMap = sparseFeatureVectors[i];

            for (const auto& pair : featureMap)
            {
                outFile << globalIndex + i << "," << pair.first << "," << pair.second << "\n";
            }
        }

        outFile.close();

        if (outFile.fail())
        {
            throw std::runtime_error("Failed to write or close the file: " + filename);
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("An error occurred while writing to the file: " + std::string(e.what()));
    }
}

#endif //CSVFILEHANDLER_HPP
