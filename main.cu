#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include "ga_kernels.cuh"
#include <stdio.h>
#include <iostream>
#include <vector>

using namespace std;

const int NUM_CITIES = 7;

const int SIZE_INDIVIDUAL = NUM_CITIES + 1;
const int POPULATION_SIZE = 100;
const int NUM_GENERATIONS = 1;
const int TOURNAMENT_SIZE = POPULATION_SIZE;
const float MUTATION_RATE = 1.0;

int main() {

    vector<vector<int>> population = initialize_population(POPULATION_SIZE, NUM_CITIES);

    cout << "Population:" << endl;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        cout << "Permutation " << i << ": ";
        for (int city : population[i]) {
            cout << city << " ";
        }
        cout << endl;
    }

    int *d_population, *d_selected_individ, *d_offspring;
    float *d_fitness_scores, *d_distance_matrix;
    curandState *d_states;

    cudaMalloc(&d_population, POPULATION_SIZE * SIZE_INDIVIDUAL * sizeof(int));
    cudaMalloc(&d_selected_individ, POPULATION_SIZE * SIZE_INDIVIDUAL * sizeof(int));
    cudaMalloc(&d_offspring, POPULATION_SIZE * SIZE_INDIVIDUAL * sizeof(int));
    cudaMalloc(&d_fitness_scores, POPULATION_SIZE * sizeof(float));
    cudaMalloc(&d_distance_matrix, NUM_CITIES * NUM_CITIES * sizeof(float));
    cudaMalloc(&d_states, POPULATION_SIZE * sizeof(curandState));

    vector<int> flat_population;
    flat_population.reserve(POPULATION_SIZE * SIZE_INDIVIDUAL);
    for (const auto& individual : population) {
        flat_population.insert(flat_population.end(), individual.begin(), individual.end());
    }

    /*cout << "flat_population: ";
    for (const int& element : flat_population) {
        cout << element << " ";
    }
    cout << endl;*/

    cudaMemcpy(d_population, flat_population.data(), POPULATION_SIZE * SIZE_INDIVIDUAL * sizeof(int), cudaMemcpyHostToDevice);

    float distance_matrix[NUM_CITIES * NUM_CITIES];
    generate_distance_matrix(distance_matrix, NUM_CITIES);

    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            cout << distance_matrix[i * NUM_CITIES + j] << "\t";
        }
        cout << endl;
    }
    
    cudaMemcpy(d_distance_matrix, distance_matrix, NUM_CITIES * NUM_CITIES * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (POPULATION_SIZE + threads_per_block - 1) / threads_per_block;

    //setup kernel
    setup_kernel << <num_blocks, threads_per_block >> > (d_states, time(NULL));
    
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        evaluate_population << <num_blocks, threads_per_block >> > (d_population, d_fitness_scores, d_distance_matrix, POPULATION_SIZE, SIZE_INDIVIDUAL, NUM_CITIES);
    
        //tournament_selection << <num_blocks, threads_per_block >> > (d_population, d_selected_individ, d_fitness_scores, POPULATION_SIZE, SIZE_INDIVIDUAL, NUM_CITIES, TOURNAMENT_SIZE, d_states);

        simple_mutation << <num_blocks, threads_per_block >> > (d_population, d_offspring, POPULATION_SIZE, SIZE_INDIVIDUAL, NUM_CITIES, MUTATION_RATE, d_states);
    }
    
    vector<int> selected_individ_flat(POPULATION_SIZE * SIZE_INDIVIDUAL);
    //cudaMemcpy(selected_individ_flat.data(), d_selected_individ, POPULATION_SIZE * SIZE_INDIVIDUAL * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(selected_individ_flat.data(), d_offspring, POPULATION_SIZE * SIZE_INDIVIDUAL * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "size of flat result: " << selected_individ_flat.size() << endl;
    for (int i = 0; i < selected_individ_flat.size(); ++i) {
        cout << selected_individ_flat[i] << " ";
        if ((i + 1) % SIZE_INDIVIDUAL == 0) {
            cout << endl;
        }
    }
    
    vector<float> fitness_scores(POPULATION_SIZE);
    cudaMemcpy(fitness_scores.data(), d_fitness_scores, POPULATION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        cout << "Fitness " << i << " : " << fitness_scores[i] << "\n";
    }
    
    cudaFree(d_population);
    cudaFree(d_offspring);
    cudaFree(d_selected_individ);

    cudaFree(d_fitness_scores);    
    cudaFree(d_distance_matrix);

    cudaFree(d_states);

    return 0;
}
