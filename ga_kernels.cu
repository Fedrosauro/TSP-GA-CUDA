#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>

using namespace std;

__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void evaluate_population(int* population, float* fitness_scores, float* distance_matrix, int num_individuals, int size_individual, int num_cities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_individuals) {
        float fitness = 0.0;
        for (int i = 0; i < size_individual - 1; ++i) {
            int idx_city1 = population[idx * size_individual + i];
            int idx_city2 = population[idx * size_individual + i + 1];
            fitness += distance_matrix[idx_city1 * size_individual + idx_city2];
        }

        fitness_scores[idx] = fitness;
    }
}

__global__ void tournament_selection(int* population, int* selected_parents, float* fitness_scores, int num_individuals, int size_individual, int num_cities, int tournament_size, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_individuals) {
        float best_fitness = FLT_MAX;
        int best_idx = -1;

        for (int i = 0; i < tournament_size; ++i) {
            int competitor_idx = curand(&states[idx]) % num_individuals;
            if ((fitness_scores[competitor_idx] < best_fitness) || best_idx == -1) {
                best_fitness = fitness_scores[competitor_idx];
                best_idx = competitor_idx;
            }
        }

        for (int j = 0; j < size_individual; ++j) {
            selected_parents[idx * size_individual + j] = population[best_idx * size_individual + j];
        }

    }
}

__global__ void simple_mutation(int* selected_parents, int* offspring, int num_individuals, int size_individual, int num_cities, float mutation_rate, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_individuals) {
        if (curand_uniform(&states[idx]) < mutation_rate) {
            int idx_city1 = ((curand(&states[idx]) % (num_cities - 1))) + 1;
            int idx_city2 = ((curand(&states[idx]) % (num_cities - 1))) + 1;
            
            for (int j = 0; j < size_individual; ++j) {
                if (j != idx_city1 && j != idx_city2) {
                    offspring[idx * size_individual + j] = selected_parents[idx * size_individual + j];
                }
                else {
                    if (j == idx_city1) {
                        offspring[idx * size_individual + idx_city1] = selected_parents[idx * size_individual + idx_city2];
                    }
                    if (j == idx_city2) {
                        offspring[idx * size_individual + idx_city2] = selected_parents[idx * size_individual + idx_city1];
                    }
                }
            }
        }
    }
}


__global__ void crossover(int* selected_parents, int* offspring, int num_individuals, int size_individual, int num_cities, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_individuals / 2) {
        int parent1_idx = 2 * idx;
        int parent2_idx = 2 * (idx + 1);

        int idx_city_crossover = ((curand(&states[idx]) % (num_cities - 1))) + 1;

        for (int j = 0; j < size_individual; ++j) {
            if (j <= idx_city_crossover) {
                offspring[parent1_idx * size_individual + j] = selected_parents[parent2_idx * size_individual + j];
                offspring[parent2_idx * size_individual + j] = selected_parents[parent1_idx * size_individual + j];
            }
            else {
                offspring[parent1_idx * size_individual + j] = selected_parents[parent1_idx * size_individual + j];
                offspring[parent2_idx * size_individual + j] = selected_parents[parent2_idx * size_individual + j];
            }
        }



    }

    
}

/*

    0 1 2 3 4 5 6 0
    0 4 3 1 5 6 2 0

*/
