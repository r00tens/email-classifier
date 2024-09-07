#ifndef TEXTPROCESSOR_HPP
#define TEXTPROCESSOR_HPP

#include <string>
#include <unordered_map>

class TextProcessor
{
public:
    static void extractTextsAndLabels(const std::vector<std::vector<std::string>>& data,
                                      std::vector<std::string>& texts, std::vector<int>& labels);
    static void printTextsAndLabels(const std::vector<std::string>& texts, const std::vector<int>& labels);
    static auto toLowercase(const std::string& text) -> std::string;
    static auto removePunctuationAndSpecialChars(const std::string& text) -> std::string;
    static auto removeStopWords(const std::string& text) -> std::string;
    static auto cleanText(const std::string& text) -> std::string;
    static auto tokenize(const std::string& text) -> std::vector<std::string>;
    static auto processText(const std::string& text) -> std::vector<std::string>;
    static auto countWordFrequency(const std::vector<std::string>& texts) -> std::unordered_map<std::string, int>;
    static void filterRareWords(std::unordered_map<std::string, int>& vocabulary,
                                const std::unordered_map<std::string, int>& wordCount, int minFrequency);
    static void buildVocabulary(const std::vector<std::string>& texts,
                                std::unordered_map<std::string, int>& vocabulary);
    static void printVocabulary(const std::unordered_map<std::string, int>& vocabulary);
    static auto textToSparseFeatureVector(const std::unordered_map<std::string, int>& vocabulary,
                                          const std::string& text) -> std::unordered_map<int, int>;
    static auto createSparseFeatureVectors(const std::unordered_map<std::string, int>& vocabulary,
                                           const std::vector<std::string>& texts) -> std::vector<std::unordered_map<
        int, int>>;
};

#endif //TEXTPROCESSOR_HPP
