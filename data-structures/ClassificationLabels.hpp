#ifndef CLASSIFICATIONLABELS_HPP
#define CLASSIFICATIONLABELS_HPP

#include <vector>

struct ClassificationLabels
{
    std::vector<int> predictedLabels;
    std::vector<int> trueLabels;
};

#endif //CLASSIFICATIONLABELS_HPP
