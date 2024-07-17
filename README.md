# TSP Problem - Genetic Algorithm and Cuda Parallelization implementation

## Overview

This project demonstrates the use of Genetic Algorithms (GA) to solve the Travelling Salesman Problem (TSP). The TSP is a classic optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the origin city. To enhance performance, this implementation leverages CUDA for parallel computing.

## File Descriptions

### Project Files
- **TSP_GA_CUDA.sln**: The Visual Studio solution file for the project.
- **TSP_GA_CUDA.vcxproj**: The Visual Studio project file which contains the configuration for building the project.
- **TSP_GA_CUDA.vcxproj.filters**: This file contains filters for organizing the project files in Visual Studio.

### Source Files
- **ga_kernels.cu**: CUDA kernels implementing the genetic algorithm operations such as selection, crossover, and mutation.
- **ga_kernels.cuh**: Header file for `ga_kernels.cu`, declaring the CUDA kernel functions.
- **main.cu**: The main file that initializes the TSP problem, sets up the GA, and handles the execution flow of the program.
- **utils.cpp**: Contains utility functions used throughout the project.
- **utils.h**: Header file for `utils.cpp`, declaring utility functions.

## Getting Started

### Prerequisites

- [Visual Studio](https://visualstudio.microsoft.com/) with CUDA toolkit installed.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### Building the Project

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/TSP_GA_CUDA.git
2. Open the solution file **TSP_GA_CUDA.sln** in Visual Studio
3. Build the solution by **selecting Build > Build Solution** or pressing Ctrl+Shift+B

### Running the Project

After building the project, you can run the executable generated in the x64/Debug or x64/Release directory, depending on your build configuration. Or you can execute it directly on VS
