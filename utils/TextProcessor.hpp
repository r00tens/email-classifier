#ifndef TEXTPROCESSOR_HPP
#define TEXTPROCESSOR_HPP

#include <string>
#include <unordered_map>

class TextProcessor
{
public:
    TextProcessor();
    ~TextProcessor();

    static void extract_texts_and_labels(const std::vector<std::vector<std::string>>& data,
                                         std::vector<std::string>& texts, std::vector<int>& labels);
};

#endif //TEXTPROCESSOR_HPP
