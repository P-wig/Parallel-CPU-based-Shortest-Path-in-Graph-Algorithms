/*
Compile: g++ -fopenmp -O3 -I../.. -static -o delta-stepping_omp delta-stepping_omp.cpp

Run: ./delta-stepping_omp internet.egr [output_file]
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <climits>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include "ECLgraph.h"

// Parallel Delta-Stepping using OpenMP
void delta_stepping_omp(const ECLgraph& g, int source, int* dist, int delta) {
    int n = g.nodes;
    
    // Initialize distances
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dist[i] = INT_MAX;
    }
    dist[source] = 0;

    // Create buckets (using sets for automatic sorting and uniqueness)
    int num_buckets = n + 1;
    std::vector<std::set<int>> buckets(num_buckets);
    buckets[0].insert(source);

    int current_bucket = 0;
    while (current_bucket < num_buckets) {
        // Skip empty buckets
        if (buckets[current_bucket].empty()) {
            current_bucket++;
            continue;
        }

        std::set<int> S;
        
        // Light edge relaxation phase
        while (!buckets[current_bucket].empty()) {
            // Move current bucket to S
            S.insert(buckets[current_bucket].begin(), buckets[current_bucket].end());
            buckets[current_bucket].clear();

            // Process light edges in parallel
            std::vector<int> S_vec(S.begin(), S.end());
            #pragma omp parallel for schedule(dynamic)
            for (size_t idx = 0; idx < S_vec.size(); idx++) {
                int u = S_vec[idx];
                if (dist[u] == INT_MAX) continue;
                
                int start = g.nindex[u];
                int end = g.nindex[u + 1];
                
                for (int i = start; i < end; i++) {
                    int v = g.nlist[i];
                    int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                    
                    // Only process light edges
                    if (weight <= delta) {
                        int new_dist = dist[u] + weight;
                        
                        #pragma omp critical
                        {
                            if (new_dist < dist[v]) {
                                dist[v] = new_dist;
                                int b = new_dist / delta;
                                if (b < num_buckets) {
                                    buckets[b].insert(v);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Heavy edge relaxation phase
        std::vector<int> S_vec(S.begin(), S.end());
        #pragma omp parallel for schedule(dynamic)
        for (size_t idx = 0; idx < S_vec.size(); idx++) {
            int u = S_vec[idx];
            if (dist[u] == INT_MAX) continue;
            
            int start = g.nindex[u];
            int end = g.nindex[u + 1];
            
            for (int i = start; i < end; i++) {
                int v = g.nlist[i];
                int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                
                // Only process heavy edges
                if (weight > delta) {
                    int new_dist = dist[u] + weight;
                    
                    #pragma omp critical
                    {
                        if (new_dist < dist[v]) {
                            dist[v] = new_dist;
                            int b = new_dist / delta;
                            if (b < num_buckets) {
                                buckets[b].insert(v);
                            }
                        }
                    }
                }
            }
        }

        current_bucket++;
    }
}

int main(int argc, char* argv[]) {
    // check command line
    if (argc != 2 && argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file [output_file]\n";
        exit(-1);
    }

    std::string output_file = (argc == 3)
        ? std::string("results/") + argv[2]
        : "results/delta_stepping_omp_results.txt";

    // read input
    ECLgraph g = readECLgraph(argv[1]);
    std::cout << "input: " << argv[1] << "\n";
    std::cout << "output: " << output_file << "\n";
    std::cout << "nodes: " << g.nodes << "\n";
    std::cout << "edges: " << g.edges << "\n";

    if (g.eweight != NULL) {
        std::cout << "graph has edge weights\n";
    } else {
        std::cout << "graph has no edge weights (using weight = 1)\n";
    }

    // Calculate average edge weight for delta parameter
    int average_edge_weight = 1;
    if (g.eweight != NULL && g.edges > 0) {
        long long sum_weights = 0;
        for (int i = 0; i < g.edges; i++) {
            sum_weights += g.eweight[i];
        }
        average_edge_weight = static_cast<int>(sum_weights / g.edges);
    }
    int delta = std::max(1, average_edge_weight);

    std::cout << "delta (bucket width) chosen: " << delta << "\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";

    // allocate distance array
    int* dist = new int[g.nodes];

    // start time
    auto beg = std::chrono::high_resolution_clock::now();

    // execute timed code - compute shortest paths from node 0
    int source = 0;
    delta_stepping_omp(g, source, dist, delta);

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

    std::cout << "\ncompute time: " << runtime.count() << " s\n";
    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Global max shortest-path: ";
    if (global_max_path == INT_MIN) {
        std::cout << "None found\n";
    } else {
        std::cout << global_max_path << "\n";
    }

    // clean up
    delete[] dist;
    freeECLgraph(g);
    return 0;
}