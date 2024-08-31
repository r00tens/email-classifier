#ifndef CSVFILEHANDLER_HPP
#define CSVFILEHANDLER_HPP

#include <string>
#include <unordered_map>
#include <vector>

class CsvFileHandler
{
public:
    CsvFileHandler();
    ~CsvFileHandler();

    static std::vector<std::vector<std::string>> read_data(const std::string& filename);
    static void print_read_data(const std::vector<std::vector<std::string>>& data);
    static void write_data(const std::vector<std::vector<std::string>>& data, const std::string& output_filename);
    static void write_data(const std::unordered_map<std::string, int>&  data, const std::string& output_filename);
    static void write_data(const std::vector<std::unordered_map<int, int>>&  data, const std::string& output_filename);
};

#endif //CSVFILEHANDLER_HPP
