#include <vector>
#include <random>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include "utils.h"

using namespace std;

vector<int> create_individual(int num_cities) {
    vector<int> individual(num_cities - 1);
    iota(individual.begin(), individual.end(), 1);  // Fill with 1 to num_cities-1
    random_shuffle(individual.begin(), individual.end());

    // Insert 0 at the beginning and end
    individual.insert(individual.begin(), 0);
    individual.push_back(0);

    return individual;
}

vector<vector<int>> initialize_population(int population_size, int num_cities) {
    vector<vector<int>> population(population_size);
    for (int i = 0; i < population_size; ++i) {
        population[i] = create_individual(num_cities);
    }
    return population;
}

void generate_distance_matrix(float* distance_matrix, int num_cities) {
    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < num_cities; ++i) {
        for (int j = 0; j < num_cities; ++j) {
            if (i == j) {
                distance_matrix[i * num_cities + j] = 0.0f;
            }
            else if (i < j) {
                float random_value = static_cast<float>(rand()) / RAND_MAX * 100.0f;
                distance_matrix[i * num_cities + j] = random_value;
                distance_matrix[j * num_cities + i] = random_value;
            }
        }
    }
}