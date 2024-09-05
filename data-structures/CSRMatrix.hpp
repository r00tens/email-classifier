#ifndef CSRMATRIX_HPP
#define CSRMATRIX_HPP

struct CSRMatrix
{
    std::vector<size_t> rowPointers;
    std::vector<size_t> columnIndices;
    std::vector<int> values;
};

inline auto convertMapToCSR(const std::vector<std::unordered_map<int, int>>& sparseFeatureVectors) -> CSRMatrix
{
    CSRMatrix csr;
    size_t nonZeroCount = 0;

    csr.rowPointers.push_back(0);

    for (const auto& row : sparseFeatureVectors)
    {
        for (const auto& [col, value] : row)
        {
            csr.columnIndices.push_back(col);
            csr.values.push_back(value);
            nonZeroCount++;
        }

        csr.rowPointers.push_back(nonZeroCount);
    }

    return csr;
}

inline auto loadSparseFeatureVectorsToCSR(const std::string& filename) -> CSRMatrix
{
    CSRMatrix csr;

    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    int currentRow = -1;

    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream sss(line);
        int vectorIndex;
        int key;
        int value;
        char delimiter;

        if (sss >> vectorIndex >> delimiter >> key >> delimiter >> value)
        {
            if (vectorIndex != currentRow)
            {
                for (int i = currentRow + 1; i <= vectorIndex; ++i)
                {
                    csr.rowPointers.push_back(csr.columnIndices.size());
                }
                currentRow = vectorIndex;
            }

            csr.columnIndices.push_back(key);
            csr.values.push_back(value);
        }
    }

    csr.rowPointers.push_back(csr.columnIndices.size());

    file.close();

    return csr;
}

#endif // CSRMATRIX_HPP
