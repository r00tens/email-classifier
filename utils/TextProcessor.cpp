#include "TextProcessor.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>

void TextProcessor::extractTextsAndLabels(const std::vector<std::vector<std::string>>& data,
                                          std::vector<std::string>& texts, std::vector<int>& labels)
{
    bool isFirstRow = true;

    for (const auto& row : data)
    {
        if (isFirstRow)
        {
            isFirstRow = false;

            continue;
        }

        if (row.size() > 1)
        {
            texts.push_back(row[1]);

            if ((row[0] == "spam" || row[0] == "Spam") || (row[0] == "1"))
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
            throw std::runtime_error("Invalid row encountered in data extraction: row has less than 2 columns.");
        }
    }
}

void TextProcessor::printTextsAndLabels(const std::vector<std::string>& texts, const std::vector<int>& labels)
{
    std::cout << "Texts size: " << texts.size() << '\n';

    for (unsigned int idx{1}; const auto& text : texts)
    {
        std::cout << idx++ << ": " << text << '\n';
    }

    std::cout << "Labels size: " << labels.size() << '\n';

    for (unsigned int idx{1}; const auto& label : labels)
    {
        std::cout << idx++ << ": " << label << '\n';
    }
}

auto TextProcessor::toLowercase(const std::string& text) -> std::string
{
    if (text.empty())
    {
        throw std::invalid_argument("to_lowercase: Input text is empty.");
    }

    std::string result = text;
    std::ranges::transform(result, result.begin(), ::tolower);

    return result;
}

auto TextProcessor::removePunctuationAndSpecialChars(const std::string& text) -> std::string
{
    if (text.empty())
    {
        throw std::invalid_argument("remove_punctuation_and_special_chars: Input text is empty.");
    }

    std::string result = text;

    std::erase_if(result, [](const unsigned char chr)
    {
        return std::ispunct(chr) || (!std::isalnum(chr) && !std::isspace(chr));
    });

    return result;
}

auto TextProcessor::removeStopWords(const std::string& text) -> std::string
{
    const std::unordered_set<std::string> stopWords = {
        // "i", "you", "he", "she", "it", "we", "they",
        // "am", "is", "are", "was", "were", "be", "being", "been",
        // "me", "him", "her", "us", "them",
        // "my", "your", "his", "her", "its", "our", "their",
        // "and", "or", "but", "nor", "yet", "so",
        // "in", "on", "at", "by", "with", "from", "to", "of", "for", "as", "about", "against", "between", "during", "through", "over", "under", "into", "out", "up", "down",
        // "this", "that", "these", "those",
        // "who", "what", "which", "where", "when", "how",
        // "have", "has", "had",
        // "can", "could", "will", "would", "shall", "should", "may", "might", "must",
        // "the", "a", "an", "not", "no", "if", "while", "just", "even", "also", "only", "than", "very"
    };

    std::istringstream iss(text);
    std::string word;
    std::string result;

    while (iss >> word)
    {
        if (!stopWords.contains(word))
        {
            result += word + " ";
        }
    }

    if (!result.empty())
    {
        result.pop_back();
    }

    return result;
}

auto TextProcessor::cleanText(const std::string& text) -> std::string
{
    if (text.empty())
    {
        throw std::invalid_argument("clean_text: Input text is empty.");
    }

    const std::string noPuncSpec = removePunctuationAndSpecialChars(text);
    const std::string noStopWordsText = removeStopWords(noPuncSpec);

    return noStopWordsText;
}

auto TextProcessor::tokenize(const std::string& text) -> std::vector<std::string>
{
    std::istringstream iss(text);
    std::vector<std::string> tokens;
    std::string token;

    while (iss >> token)
    {
        tokens.push_back(token);
    }

    return tokens;
}

auto TextProcessor::processText(const std::string& text) -> std::vector<std::string>
{
    if (text.empty())
    {
        return {};
    }

    const std::string lowerText = toLowercase(text);
    const std::string cleanedText = cleanText(lowerText);
    const std::vector<std::string> tokens = tokenize(cleanedText);

    return tokens;
}

auto TextProcessor::countWordFrequency(const std::vector<std::string>& texts) -> std::unordered_map<std::string, int>
{
    std::unordered_map<std::string, int> wordCount;

    for (const auto& text : texts)
    {
        std::vector<std::string> tokens = processText(text);

        for (const auto& token : tokens)
        {
            wordCount[token]++;
        }
    }

    return wordCount;
}

void TextProcessor::filterRareWords(std::unordered_map<std::string, int>& vocabulary,
                                    const std::unordered_map<std::string, int>& wordCount, int minFrequency)
{
    for (auto it = vocabulary.begin(); it != vocabulary.end();)
    {
        if (wordCount.at(it->first) < minFrequency)
        {
            it = vocabulary.erase(it); // Usunięcie słów, które pojawiają się rzadziej niż min_frequency
        }
        else
        {
            ++it;
        }
    }
}

void TextProcessor::buildVocabulary(const std::vector<std::string>& texts,
                                    std::unordered_map<std::string, int>& vocabulary)
{
    int index{};
    std::unordered_map<std::string, int> wordCount = countWordFrequency(texts);

    constexpr int MIN_FREQUENCY = 3;

    for (const auto& text : texts)
    {
        std::vector<std::string> tokens = processText(text);

        for (const auto& token : tokens)
        {
            if (!vocabulary.contains(token) && wordCount[token] >= MIN_FREQUENCY)
            {
                vocabulary[token] = index++;
            }
        }
    }
}

void TextProcessor::printVocabulary(const std::unordered_map<std::string, int>& vocabulary)
{
    std::cout << "Vocabulary size: " << vocabulary.size() << '\n';

    for (unsigned int idx{1}; const auto& [word, index] : vocabulary)
    {
        std::cout << idx++ << ": " << word << " -> " << index << '\n';
    }
}

auto TextProcessor::textToSparseFeatureVector(
    const std::unordered_map<std::string, int>& vocabulary, const std::string& text) -> std::unordered_map<int, int>
{
    std::unordered_map<int, int> featureVector;
    std::vector<std::string> tokens = processText(text);

    for (const auto& token : tokens)
    {
        auto iter = vocabulary.find(token);

        if (iter != vocabulary.end())
        {
            featureVector[iter->second]++;
        }
    }

    return featureVector;
}

auto TextProcessor::createSparseFeatureVectors(
    const std::unordered_map<std::string, int>& vocabulary,
    const std::vector<std::string>& texts) -> std::vector<std::unordered_map<int, int>>
{
    std::vector<std::unordered_map<int, int>> featureVectors;

    for (const auto& text : texts)
    {
        featureVectors.push_back(textToSparseFeatureVector(vocabulary, text));
    }

    return featureVectors;
}
