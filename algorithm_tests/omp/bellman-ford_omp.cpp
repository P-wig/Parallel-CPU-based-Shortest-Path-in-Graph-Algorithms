/*
Compile: g++ -fopenmp -O3 -I../.. -static -o bellman-ford_omp bellman-ford_omp.cpp

Run: ./bellman-ford_omp internet.egr [output_file]
*/

#include <iostream>
#include <fstream>
#include <climits>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include "ECLgraph.h"

// Helper function for atomic minimum
inline int atomic_min(int* addr, int value) {
    int old = *addr;
    while (value < old) {
        int prev;
        #pragma omp atomic capture
        { prev = *addr; *addr = std::min(*addr, value); }
        if (prev <= value) break;
        old = prev;
    }
    return old;
}

// Parallel Bellman-Ford using OpenMP (optimized)
bool bellman_ford_omp(const ECLgraph& g, int source, int* dist) {
    int n = g.nodes;
    
    // Initialize distances
    #pragma omp parallel for
    for (int i = 0; i < g.nodes; i++) {
        dist[i] = INT_MAX;
    }
    dist[source] = 0;

    // Relax edges |V|-1 times
    for (int iter = 1; iter < n; iter++) {
        bool changed = false;
        
        // Create a temporary array for this iteration
        int* new_dist = new int[n];
        
        // Copy current distances
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            new_dist[i] = dist[i];
        }
        
        // Parallelize edge relaxation across all nodes
        #pragma omp parallel for schedule(dynamic, 256) reduction(||:changed)
        for (int u = 0; u < n; u++) {
            if (dist[u] != INT_MAX) {
                int start = g.nindex[u];
                int end = g.nindex[u + 1];
                for (int i = start; i < end; i++) {
                    int v = g.nlist[i];
                    int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                    int new_dist_val = dist[u] + weight;
                    
                    if (new_dist_val < new_dist[v]) {
                        int old = atomic_min(&new_dist[v], new_dist_val);
                        if (new_dist_val < old) {
                            changed = true;
                        }
                    }
                }
            }
        }
        
        // Copy new distances back
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            dist[i] = new_dist[i];
        }
        
        delete[] new_dist;
        
        // Early exit if no changes
        if (!changed) break;
    }

    return true;
}

int main(int argc, char* argv[]) {
    // check command line
    if (argc != 3 && argc != 4) {
        std::cerr << "USAGE: " << argv[0] << " input_file num_threads [output_file]\n";
        exit(-1);
    }

    // read input
    ECLgraph g = readECLgraph(argv[1]);

    // Set number of threads
    int num_threads = atoi(argv[2]);
    if (num_threads < 1) {
        std::cerr << "ERROR: num_threads must be at least 1\n";
        exit(-1);
    }
    omp_set_num_threads(num_threads);

    std::string output_file = (argc == 4)
        ? std::string("results/") + argv[3]
        : "results/bellman_ford_omp_results.txt";
    std::string console_file = "results/bellman-ford_omp_" + std::to_string(num_threads) + "_results.txt";
    std::ofstream console_out(console_file);
    if (!console_out) {
        std::cerr << "ERROR: could not open console output file\n";
        exit(-1);
    }

    console_out << "input: " << argv[1] << "\n";
    console_out << "output: " << output_file << "\n";
    console_out << "nodes: " << g.nodes << "\n";
    console_out << "edges: " << g.edges << "\n";

    if (g.eweight != NULL) {
        console_out << "graph has edge weights\n";
    } else {
        console_out << "graph has no edge weights (using weight = 1)\n";
    }

    console_out << "OpenMP threads: " << omp_get_max_threads() << "\n";

    // allocate distance array
    int* dist = new int[g.nodes];

    // start time
    auto beg = std::chrono::high_resolution_clock::now();

    // execute timed code - compute shortest paths from node 0
    int source = 0;
    bool ok = bellman_ford_omp(g, source, dist);

    // end time
    auto end = std::chrono::high_resolution_clock::now();

    // calc
    std::chrono::duration<double> runtime = end - beg;

    // Write results to file
    std::ofstream outfile(output_file);
    if (!outfile) {
        std::cerr << "ERROR: could not open output file\n";
        delete[] dist;
        freeECLgraph(g);
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
            if (dist[i] > global_max_path) {
                global_max_path = dist[i];
            }
        }
    }
    outfile.close();

    if (!ok) console_out << "WARNING: Negative-weight cycle detected!\n";
    console_out << "\ncompute time: " << runtime.count() << " s\n";
    console_out << "Results written to " << output_file << "\n";
    console_out << "Global max shortest-path: ";
    if (global_max_path == INT_MIN) {
        console_out << "None found\n";
    } else {
        console_out << global_max_path << "\n";
    }
    console_out.close();

    // clean up
    delete[] dist;
    freeECLgraph(g);
    return 0;
}