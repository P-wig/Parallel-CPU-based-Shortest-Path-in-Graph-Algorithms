/*
Compile: g++ -fopenmp -O3 -I../.. -static -o delta-stepping_omp delta-stepping_omp.cpp

Run: ./delta-stepping_omp input_file num_threads [output_file]
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

// Parallel Delta-Stepping using OpenMP with thread-local buckets
void delta_stepping_omp(const ECLgraph& g, int source, int* dist, int delta, int num_threads) {
    int n = g.nodes;
    
    // Initialize distances
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dist[i] = INT_MAX;
    }
    dist[source] = 0;

    // Calculate realistic bucket count based on max possible distance
    int max_dist = 0;
    if (g.eweight != NULL) {
        for (int i = 0; i < g.edges; i++) {
            max_dist += g.eweight[i];
        }
    } else {
        max_dist = g.edges;
    }
    int bucket_count = max_dist / delta + 2;

    // Create global buckets and thread-local buckets
    std::vector<std::set<int>> buckets(bucket_count);
    std::vector<std::vector<std::set<int>>> local_buckets(num_threads);
    for (int t = 0; t < num_threads; t++) {
        local_buckets[t].resize(bucket_count);
    }
    buckets[0].insert(source);

    int current_bucket = 0;
    int bucket_frontier = 0;
    
    while (current_bucket < bucket_count) {
        // Skip empty buckets
        if (buckets[current_bucket].empty()) {
            current_bucket++;
            continue;
        }

        std::set<int> S_global;
        
        // Light edge relaxation phase
        while (!buckets[current_bucket].empty()) {
            // Move current bucket to temporary vector
            std::vector<int> curr_nodes(buckets[current_bucket].begin(), buckets[current_bucket].end());
            buckets[current_bucket].clear();
            bool changed = false;

            // Process light edges in parallel
            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                std::set<int> local_S;
                
                #pragma omp for schedule(dynamic) reduction(||:changed)
                for (size_t idx = 0; idx < curr_nodes.size(); idx++) {
                    int u = curr_nodes[idx];
                    local_S.insert(u);
                    if (dist[u] == INT_MAX) continue;
                    
                    int start = g.nindex[u];
                    int end = g.nindex[u + 1];
                    
                    for (int i = start; i < end; i++) {
                        int v = g.nlist[i];
                        int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                        
                        // Only process light edges
                        if (weight <= delta) {
                            int new_dist = dist[u] + weight;
                            
                            if (new_dist < dist[v]) {
                                int old = atomic_min(&dist[v], new_dist);
                                if (new_dist < old) {
                                    int b = new_dist / delta;
                                    // Apply bucket frontier
                                    if (b < bucket_frontier) b = bucket_frontier;
                                    if (b < bucket_count) {
                                        local_buckets[tid][b].insert(v);
                                        if (b == current_bucket) changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Merge local S into global S
                #pragma omp critical
                {
                    S_global.insert(local_S.begin(), local_S.end());
                }
            }

            // Merge thread-local buckets into global buckets
            for (int t = 0; t < num_threads; t++) {
                for (int b = 0; b < bucket_count; b++) {
                    buckets[b].insert(local_buckets[t][b].begin(), local_buckets[t][b].end());
                    local_buckets[t][b].clear();
                }
            }
            
            // Break if no changes or bucket is empty
            if (!changed || buckets[current_bucket].empty()) break;
        }

        // Heavy edge relaxation phase
        std::vector<int> S_vec(S_global.begin(), S_global.end());
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for schedule(dynamic)
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
                        
                        if (new_dist < dist[v]) {
                            int old = atomic_min(&dist[v], new_dist);
                            if (new_dist < old) {
                                int b = new_dist / delta;
                                // Apply bucket frontier
                                if (b < bucket_frontier) b = bucket_frontier;
                                if (b < bucket_count) {
                                    local_buckets[tid][b].insert(v);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Merge thread-local buckets after heavy edge phase
        for (int t = 0; t < num_threads; t++) {
            for (int b = 0; b < bucket_count; b++) {
                buckets[b].insert(local_buckets[t][b].begin(), local_buckets[t][b].end());
                local_buckets[t][b].clear();
            }
        }

        // Advance to next non-empty bucket
        bucket_frontier = current_bucket;
        current_bucket++;
        while (current_bucket < bucket_count && buckets[current_bucket].empty()) {
            current_bucket++;
        }
    }
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
        : "results/delta_stepping_omp_results.txt";
    std::string console_file = "results/delta-stepping_omp_" + std::to_string(num_threads) + "_results.txt";
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

    int delta = 1000;
    console_out << "delta (bucket width) chosen: " << delta << "\n";
    console_out << "OpenMP threads: " << omp_get_max_threads() << "\n";

    // allocate distance array
    int* dist = new int[g.nodes];

    // start time
    auto beg = std::chrono::high_resolution_clock::now();

    // execute timed code - compute shortest paths from node 0
    int source = 0;
    delta_stepping_omp(g, source, dist, delta, num_threads);

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