#ifndef TEXTPROCESSOR_HPP
#define TEXTPROCESSOR_HPP

#include <string>
#include <unordered_map>

class TextProcessor
{
public:
    TextProcessor();
    ~TextProcessor();

    static void extract_texts_and_labels(const std::vector<std::vector<std::string>>& data, std::vector<std::string>& texts, std::vector<int>& labels);
    static std::string to_lowercase(const std::string& text);
    static std::string remove_punctuation_and_special_chars(const std::string& text);
    static std::string remove_stop_words(const std::string& text);
    static std::string clean_text(const std::string& text);
    static std::vector<std::string> tokenize(const std::string& text);
    static std::vector<std::string> process_text(const std::string& text);
    static void build_vocabulary(const std::vector<std::string>& texts, std::unordered_map<std::string, int>& vocabulary);
};

#endif //TEXTPROCESSOR_HPP
