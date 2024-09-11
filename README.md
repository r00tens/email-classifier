# Text Classifier

## Project Overview

This project implements a text classifier that categorizes text into predefined categories (classes) using the Naive Bayes algorithm. It supports model training by constructing a dictionary (vocabulary) and feature vectors based on a labeled training dataset. The classifier evaluates the performance of the trained model, including metrics such as accuracy, precision, recall, and F1 score using a test dataset. The implementation runs on both the CPU (C++) and NVIDIA GPU (CUDA).

## Features

- Multinomial Naive Bayes Algorithm: A probabilistic algorithm used for text classification.
- Dictionary and Feature Vector Construction: Builds a dictionary from the training data and converts text into sparse feature vectors, where each sparse vector contains the number of occurrences of features (e.g., words) in the text.
- Training and evaluation: allows training the model on an annotated dataset and evaluates its performance on a test set using standard metrics such as accuracy, precision, recall, and F1 score.
- CPU and GPU Support: The implementation can run on both CPU (C++) and GPU (CUDA), utilizing NVIDIA GPUs for accelerated computations.
- Benchmarking: allows (primitive) comparison of CPU and GPU implementation performance on the same dataset.

## Prerequisites

- CMake ``3.29``
- CUDA ``12.6``
- Visual Studio ``17 2022`` or Ninja ``1.12.0``

## Installation

### Clone the Repository

```
git clone https://github.com/r00tens/text-classifier.git
cd text-classifier
```

### Build the Project:

##### Visual Studio 17 2022

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

#### Ninja

```
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE Release ..
cmake --build .
```

> [!NOTE]  
> The project has not yet been tested on any Linux distributions.

## Prepare the dataset

### The dataset should be in CSV format, where

- the first row contains column names (e.g., label, text)
- each row corresponds to a single text entry 
- columns in a row are separated by a comma

#### Example of the dataset format

```
label,text
0,This is a sample text
1,This is another sample text
```

## License

This project is licensed under the MIT License - for details, see the [LICENSE](LICENSE) file.
