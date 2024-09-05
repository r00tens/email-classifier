#include "CsvFileHandler.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

auto CsvFileHandler::readData(const std::string& filename) -> std::vector<std::vector<std::string>>
{
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;

    while (std::getline(file, line))
    {
        std::vector<std::string> row = parseCsvLine(line);
        if (!row.empty())
        {
            validateAndAddRow(data, row, line);
        }
    }

    file.close();

    return data;
}

auto CsvFileHandler::parseCsvLine(const std::string& line) -> std::vector<std::string>
{
    std::vector<std::string> row;
    std::string currentCell;
    bool inQuotes = false;

    for (const char currentChar : line)
    {
        if (currentChar == '"')
        {
            inQuotes = !inQuotes; // Toggle quotes state
        }
        else if (currentChar == ',' && !inQuotes)
        {
            row.push_back(currentCell);
            currentCell.clear();
        }
        else
        {
            currentCell += currentChar;
        }
    }

    row.push_back(currentCell); // Add the last cell

    return row;
}

void CsvFileHandler::validateAndAddRow(std::vector<std::vector<std::string>>& data, const std::vector<std::string>& row,
                                       const std::string& line)
{
    if (row.size() == 2) // Expecting 2 columns
    {
        data.push_back(row);
    }
    else
    {
        throw std::runtime_error("Incorrect number of columns in row: " + line);
    }
}
