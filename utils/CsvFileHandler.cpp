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
    std::vector<std::string> row;
    bool inQuotes = false;
    std::string current_cell;

    while (std::getline(file, line))
    {
        if (inQuotes)
        {
            current_cell += "\n" + line;
        }
        else
        {
            if (!current_cell.empty())
            {
                row.push_back(current_cell);
                current_cell.clear();
            }

            row.clear();
        }

        size_t i = 0;
        while (i < line.size())
        {
            char c = line[i];

            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                row.push_back(current_cell);
                current_cell.clear();
            }
            else
            {
                current_cell += c;
            }

            i++;
        }

        if (!inQuotes)
        {
            row.push_back(current_cell);
            current_cell.clear();

            if (row.size() == 2)
            {
                data.push_back(row);
            }
            else
            {
                std::cerr << "Warning: Incorrect number of columns in row, skipping: " << line << std::endl;
            }
        }
    }

    if (!current_cell.empty() && row.size() == 1)
    {
        row.push_back(current_cell);

        if (row.size() == 2)
        {
            data.push_back(row);
        }
    }

    file.close();

    return data;
}

void CsvFileHandler::print_read_data(const std::vector<std::vector<std::string>>& data)
{
    for (unsigned int i{1}; const auto& row : data)
    {
        std::cout << i++ << ": ";

        for (const auto& cell : row)
        {
            std::cout << cell << ' ';
        }

        std::cout << '\n';
    }
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
