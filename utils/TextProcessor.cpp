#include "TextProcessor.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>

TextProcessor::TextProcessor() = default;

TextProcessor::~TextProcessor() = default;

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
            throw std::runtime_error("Invalid row encountered in data extraction: row has less than 2 columns.");
        }
    }
}

void TextProcessor::print_texts_and_labels(const std::vector<std::string>& texts, const std::vector<int>& labels)
{
    std::cout << "Texts size: " << texts.size() << std::endl;

    for (unsigned int i{1}; const auto& text : texts)
    {
        std::cout << i++ << ": " << text << std::endl;
    }

    std::cout << "Labels size: " << labels.size() << std::endl;

    for (unsigned int i{1}; const auto& label : labels)
    {
        std::cout << i++ << ": " << label << std::endl;
    }
}

std::string TextProcessor::to_lowercase(const std::string& text)
{
    if (text.empty())
    {
        throw std::invalid_argument("to_lowercase: Input text is empty.");
    }

    std::string result = text;
    std::ranges::transform(result, result.begin(), ::tolower);

    return result;
}

std::string TextProcessor::remove_punctuation_and_special_chars(const std::string& text)
{
    if (text.empty())
    {
        throw std::invalid_argument("remove_punctuation_and_special_chars: Input text is empty.");
    }

    std::string result = text;

    std::erase_if(result, [](const unsigned char c)
    {
        return std::ispunct(c) || (!std::isalnum(c) && !std::isspace(c));
    });

    return result;
}

std::string TextProcessor::remove_stop_words(const std::string& text)
{
    const std::unordered_set<std::string> stop_words = {
        "is", "a", "of", "in", "the", "this", "and", "to", "with"
    };

    std::istringstream iss(text);
    std::string word;
    std::string result;

    while (iss >> word)
    {
        if (!stop_words.contains(word))
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

std::string TextProcessor::clean_text(const std::string& text)
{
    if (text.empty())
    {
        throw std::invalid_argument("clean_text: Input text is empty.");
    }

    const std::string no_punc_spec = remove_punctuation_and_special_chars(text);
    const std::string no_stop_words_text = remove_stop_words(no_punc_spec);

    return no_stop_words_text;
}

std::vector<std::string> TextProcessor::tokenize(const std::string& text)
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

std::vector<std::string> TextProcessor::process_text(const std::string& text)
{
    if (text.empty())
    {
        throw std::invalid_argument("process_text: Input text is empty.");
    }

    const std::string lower_text = to_lowercase(text);
    const std::string cleaned_text = clean_text(lower_text);
    const std::vector<std::string> tokens = tokenize(cleaned_text);

    return tokens;
}

void TextProcessor::build_vocabulary(const std::vector<std::string>& texts, std::unordered_map<std::string, int>& vocabulary)
{
    if (texts.empty())
    {
        throw std::invalid_argument("build_vocabulary: Input texts vector is empty.");
    }

    int index{};

    for (const auto& text : texts)
    {
        if (text.empty())
        {
            throw std::invalid_argument("build_vocabulary: One of the input texts is empty.");
        }

        std::vector<std::string> tokens = process_text(text);

        for (const auto& token : tokens)
        {
            if (!vocabulary.contains(token))
            {
                vocabulary[token] = index++;
            }
        }
    }
}

void TextProcessor::print_vocabulary(const std::unordered_map<std::string, int>& vocabulary)
{
    std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;

    for (unsigned int i{1}; const auto& [word, index] : vocabulary)
    {
        std::cout << i++ << ": " << word << " -> " << index << std::endl;
    }
}
