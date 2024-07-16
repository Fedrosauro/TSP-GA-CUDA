#pragma once
#include <vector>

using namespace std;

vector<int> create_individual(int num_cities) ;

vector<vector<int>> initialize_population(int population_size, int num_cities);

void generate_distance_matrix(float* distance_matrix, int num_cities) ;