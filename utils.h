#pragma once
#include <vector>
#include <random>

using namespace std;

vector<int> create_individual(int num_cities, mt19937& rng) ;

vector<vector<int>> initialize_population(int population_size, int num_cities);

void generate_distance_matrix(float* distance_matrix, int num_cities) ;

unsigned long generate_random_seed() ;
