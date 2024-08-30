#include "TextProcessor.hpp"

#include <sstream>
#include <unordered_set>

void TextProcessor::extract_texts_and_labels(const std::vector<std::vector<std::string>>& data, std::vector<std::string>& texts, std::vector<int>& labels)
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
