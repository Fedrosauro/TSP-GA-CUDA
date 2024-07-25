#include <vector>
#include <random>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include "utils.h"
#include <algorithm>

using namespace std;

vector<int> create_individual(int num_cities, mt19937& rng) {
    vector<int> individual(num_cities - 1);
    iota(individual.begin(), individual.end(), 1);  // Fill with 1 to num_cities-1
    shuffle(individual.begin(), individual.end(), rng);

    // Insert 0 at the beginning and end
    individual.insert(individual.begin(), 0);
    individual.push_back(0);

    return individual;
}

vector<vector<int>> initialize_population(int population_size, int num_cities) {
    vector<vector<int>> population(population_size);
    mt19937 rng(time(0));  // Seed random number generator with current time
    for (int i = 0; i < population_size; ++i) {
        population[i] = create_individual(num_cities, rng);
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

unsigned long generate_random_seed() {
    random_device rd;  // Non-deterministic random number generator
    mt19937_64 gen(rd());  // Standard Mersenne Twister engine
    uniform_int_distribution<unsigned long> dis;  // Uniform distribution over unsigned long range
    return dis(gen);  // Generate a random unsigned long
}