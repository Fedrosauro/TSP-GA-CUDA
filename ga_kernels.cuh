#pragma once
#include <curand_kernel.h>

__global__ void setup_kernel(curandState* state, unsigned long seed);

__global__ void evaluate_population(int* population, float* fitness_scores, float* distance_matrix, int num_individuals, int size_individual, int num_cities);

__global__ void tournament_selection(int* population, int* selected_parents, float* fitness_scores, int num_individuals, int size_individual, int num_cities, int tournament_size, curandState* states);

__global__ void simple_mutation(int* selected_parents, int* offspring, int num_individuals, int size_individual, int num_cities, float mutation_rate, curandState* states);

__global__ void simple_crossover(int* selected_parents, int* offspring, int num_individuals, int size_individual, int num_cities, curandState* states);

