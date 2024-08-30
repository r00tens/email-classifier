#include "CsvFileHandler.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

CsvFileHandler::CsvFileHandler() = default;

CsvFileHandler::~CsvFileHandler() = default;

std::vector<std::vector<std::string>> CsvFileHandler::read_data(const std::string& filename)
{
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open the file: " + filename);
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<std::string> row;
        std::string cell;
        bool inQuotes = false;

        for (const char c : line)
        {
            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                row.push_back(cell);
                cell.clear();
            }
            else
            {
                cell += c;
            }
        }

        row.push_back(cell);

        if (!row.empty())
        {
            data.push_back(row);
        }
    }

    file.close();

    return data;
}

void CsvFileHandler::write_data(const std::vector<std::vector<std::string>>& data, const std::string& output_filename)
{
    std::ofstream file(output_filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open the file: " + output_filename);
    }

    for (const auto& row : data)
    {
        for (const auto& cell : row)
        {
            file << cell << ',';
        }

        file << '\n';
    }

    file.close();
}

void CsvFileHandler::extract_texts_and_labels(const std::vector<std::vector<std::string>>& data, std::vector<std::string>& texts, std::vector<int>& labels)
{
    bool is_first_row = true;

    for (const auto& row : data)
    {
        if (is_first_row)
        {
            is_first_row = false;

            continue;
        }

        if (row.size() > 1)
        {
            texts.push_back(row[1]);

            if ((row[0] == "spam" || row[0] == "Spam"))
            {
                labels.push_back(1);
            }
            else
            {
                labels.push_back(0);
            }
        }
        else
        {
            throw std::runtime_error("Invalid row encountered in data extraction.");
        }
    }
}
