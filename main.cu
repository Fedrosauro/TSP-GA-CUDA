#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include "ga_kernels.cuh"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

const int NUM_CITIES = 100;

const int SIZE_INDIVIDUAL = NUM_CITIES + 1;
const int POPULATION_SIZE = 10000;
const int NUM_GENERATIONS = 1500;
const int TOURNAMENT_SIZE = 50;
const float MUTATION_RATE = 1.0;

int main() {

    vector<vector<int>> population = initialize_population(POPULATION_SIZE, NUM_CITIES);

    cout << "TSP GA CUDA IMPLEMENTATION START" << endl <<
            "================================" << endl;

    /*cout << "Population:" << endl;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        cout << "Permutation " << i << ": ";
        for (int city : population[i]) {
            cout << city << " ";
        }
        cout << endl;
    }*/

    int *d_population, *d_selected_individ, * d_offspring;
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

    /*for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            cout << distance_matrix[i * NUM_CITIES + j] << "\t";
        }
        cout << endl;
    }*/
    
    cudaMemcpy(d_distance_matrix, distance_matrix, NUM_CITIES * NUM_CITIES * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (POPULATION_SIZE + threads_per_block - 1) / threads_per_block;

    float best_fit = FLT_MAX;
    int best_gen = 0;
    vector<int> best_individual(SIZE_INDIVIDUAL);

    //setup kernel
    unsigned long random_seed = generate_random_seed();
    setup_kernel << <num_blocks, threads_per_block >> > (d_states, random_seed);
    
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        cout << "Generation: " << generation << endl;

        tournament_selection << <num_blocks, threads_per_block >> > (d_population, d_selected_individ, d_fitness_scores, POPULATION_SIZE, SIZE_INDIVIDUAL, NUM_CITIES, TOURNAMENT_SIZE, d_states);

        simple_crossover << <num_blocks, threads_per_block >> > (d_selected_individ, d_offspring, POPULATION_SIZE, SIZE_INDIVIDUAL, NUM_CITIES, d_states);

        simple_mutation << <num_blocks, threads_per_block >> > (d_offspring, d_population, POPULATION_SIZE, SIZE_INDIVIDUAL, NUM_CITIES, MUTATION_RATE, d_states);

        evaluate_population << <num_blocks, threads_per_block >> > (d_population, d_fitness_scores, d_distance_matrix, POPULATION_SIZE, SIZE_INDIVIDUAL, NUM_CITIES);

        cudaMemcpy(flat_population.data(), d_population, POPULATION_SIZE * SIZE_INDIVIDUAL * sizeof(int), cudaMemcpyDeviceToHost);

        vector<float> fitness_scores(POPULATION_SIZE);
        cudaMemcpy(fitness_scores.data(), d_fitness_scores, POPULATION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        auto min_it = min_element(fitness_scores.begin(), fitness_scores.end());
        int best_idx = distance(fitness_scores.begin(), min_it);
        float current_best_fit = *min_it;

        if (current_best_fit < best_fit) {
            best_fit = current_best_fit;
            best_gen = generation;
            for (int i = 0; i < SIZE_INDIVIDUAL; i++) {
                best_individual[i] = flat_population[best_idx * SIZE_INDIVIDUAL + i];
            }
        }
        cout << "Best individual: [ ";
        for (int gene : best_individual) {
            cout << gene << " ";
        }
        cout << "] found at gen: " << best_gen << endl << "Fitness: " << best_fit << endl;
    }
       
    cudaFree(d_population);
    cudaFree(d_offspring);
    cudaFree(d_selected_individ);

    cudaFree(d_fitness_scores);    
    cudaFree(d_distance_matrix);

    cudaFree(d_states);

    return 0;
}
