/*
Compile: g++ -std=c++17 -O2 -pthread -I. -o algorithm_tests/pthread/bellman-ford_pthread algorithm_tests/pthread/bellman-ford_pthread.cpp
Run: ./algorithm_tests/pthread/bellman-ford_pthread internet.egr 4
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <chrono>
#include <pthread.h>
#include <cstdlib>
#include "ECLgraph.h"

struct ThreadData {
    const ECLgraph* g;
    int* dist;
    int thread_id;
    int num_threads;
    char* changed;
};

void* relax_edges(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    const ECLgraph& g = *(data->g);
    int* dist = data->dist;
    int thread_id = data->thread_id;
    int num_threads = data->num_threads;
    char local_changed = 0;
    for (int u = thread_id; u < g.nodes; u += num_threads) {
        int start = g.nindex[u];
        int end = g.nindex[u + 1];
        for (int i = start; i < end; i++) {
            int v = g.nlist[i];
            int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                local_changed = 1;
            }
        }
    }
    *(data->changed) = local_changed;
    return nullptr;
}

int main(int argc, char* argv[]) {
    std::cout << "Single-Source Shortest Path using Bellman-Ford with pthreads\n";

    if (argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file num_threads\n";
        return -1;
    }

    int num_threads = std::atoi(argv[2]);
    if (num_threads < 1) {
        std::cerr << "ERROR: num_threads must be at least 1\n";
        return -1;
    }

    std::string output_file = "algorithm_tests/pthread/bellman_ford_pthread_results.txt";

    ECLgraph g = readECLgraph(argv[1]);
    std::cout << "input: " << argv[1] << "\n";
    std::cout << "output: " << output_file << "\n";
    std::cout << "nodes: " << g.nodes << "\n";
    std::cout << "edges: " << g.edges << "\n";
    if (g.eweight != NULL) std::cout << "graph has edge weights\n";
    else std::cout << "graph has no edge weights (using weight = 1)\n";
    std::cout << "pthreads used: " << num_threads << "\n";

    int* dist = new int[g.nodes];
    for (int i = 0; i < g.nodes; i++) dist[i] = INT_MAX;
    int source = 0;
    dist[source] = 0;

    auto beg = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter < g.nodes; iter++) {
        std::vector<pthread_t> threads(num_threads);
        std::vector<ThreadData> thread_data(num_threads);
        std::vector<char> changed(num_threads, 0); // vector of chars

        for (int t = 0; t < num_threads; t++) {
            thread_data[t].g = &g;
            thread_data[t].dist = dist;
            thread_data[t].thread_id = t;
            thread_data[t].num_threads = num_threads;
            thread_data[t].changed = &changed[t]; // pointer to char
            pthread_create(&threads[t], nullptr, relax_edges, &thread_data[t]);
        }
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], nullptr);
        }
        bool any_changed = false;
        for (int t = 0; t < num_threads; t++) {
            if (changed[t]) {
                any_changed = true;
                break;
            }
        }
        if (!any_changed) break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - beg;
    std::cout << "\ncompute time: " << runtime.count() << " s\n";

    std::ofstream outfile(output_file);
    if (!outfile) {
        std::cerr << "ERROR: could not open output file\n";
        return -1;
    }

    outfile << "# Single-Source Shortest Path from node " << source << "\n";
    outfile << "# Format: target_node distance\n";

    int global_max_path = INT_MIN;
    for (int i = 0; i < g.nodes; i++) {
        if (dist[i] == INT_MAX) {
            outfile << i << " INF\n";
        } else {
            outfile << i << " " << dist[i] << "\n";
            if (dist[i] > global_max_path) global_max_path = dist[i];
        }
    }
    outfile.close();

    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Global max shortest-path: ";
    if (global_max_path == INT_MIN) std::cout << "None found\n";
    else std::cout << global_max_path << "\n";

    delete[] dist;
    freeECLgraph(g);
    return 0;
}