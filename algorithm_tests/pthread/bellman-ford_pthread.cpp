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

static int threads;
static pthread_barrier_t barrier;
static ECLgraph g;
static int* dist;
static char* changed;
static int source = 0;
static volatile bool go_again;

void* bellman_ford_thread(void* arg) {
    const int tid = (int)(intptr_t)arg;
    const int block = g.nodes / threads;
    const int start = tid * block;
    const int stop = (tid == threads - 1) ? g.nodes : (tid + 1) * block;

    for (int iter = 1; iter < g.nodes; iter++) {
        char local_changed = 0;
        for (int u = start; u < stop; u++) {
            int start_e = g.nindex[u];
            int end_e = g.nindex[u + 1];
            for (int i = start_e; i < end_e; i++) {
                int v = g.nlist[i];
                int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    local_changed = 1;
                }
            }
        }
        changed[tid] = local_changed;

        pthread_barrier_wait(&barrier);

        // Only thread 0 checks for convergence and sets the flag
        if (tid == 0) {
            go_again = false;
            for (int t = 0; t < threads; t++) {
                if (changed[t]) {
                    go_again = true;
                    break;
                }
            }
        }

        pthread_barrier_wait(&barrier);

        // All threads check the shared flag
        if (!go_again) break;
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    std::cout << "Single-Source Shortest Path using Bellman-Ford with pthreads\n";

    if (argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file num_threads\n";
        return -1;
    }

    threads = std::atoi(argv[2]);
    if (threads < 1) {
        std::cerr << "ERROR: num_threads must be at least 1\n";
        return -1;
    }

    std::string output_file = "algorithm_tests/pthread/results/bellman_ford_pthread_results.txt";
    std::string console_file = "results/bellman-ford_pthread_" + std::to_string(threads) + "_results.txt";
    std::ofstream console_out(console_file);
    if (!console_out) {
        std::cerr << "ERROR: could not open console output file\n";
        return -1;
    }

    g = readECLgraph(argv[1]);
    console_out << "Single-Source Shortest Path using Bellman-Ford with pthreads\n";
    console_out << "input: " << argv[1] << "\n";
    console_out << "output: " << output_file << "\n";
    console_out << "nodes: " << g.nodes << "\n";
    console_out << "edges: " << g.edges << "\n";
    if (g.eweight != NULL) console_out << "graph has edge weights\n";
    else console_out << "graph has no edge weights (using weight = 1)\n";
    console_out << "pthreads used: " << threads << "\n";

    dist = new int[g.nodes];
    changed = new char[threads];
    for (int i = 0; i < g.nodes; i++) dist[i] = INT_MAX;
    dist[source] = 0;

    pthread_barrier_init(&barrier, nullptr, threads);

    // Create threads before timing
    pthread_t* handles = new pthread_t[threads - 1];

    auto beg = std::chrono::high_resolution_clock::now();

    // Launch worker threads
    for (int t = 0; t < threads - 1; t++) {
        pthread_create(&handles[t], nullptr, bellman_ford_thread, (void*)(intptr_t)t);
    }
    // Main thread also participates
    bellman_ford_thread((void*)(intptr_t)(threads - 1));

    // Join worker threads
    for (int t = 0; t < threads - 1; t++) {
        pthread_join(handles[t], nullptr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - beg;
    console_out << "\ncompute time: " << runtime.count() << " s\n";

    std::ofstream outfile(output_file);
    if (!outfile) {
        std::cerr << "ERROR: could not open output file\n";
        console_out.close();
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

    console_out << "Results written to " << output_file << "\n";
    console_out << "Global max shortest-path: ";
    if (global_max_path == INT_MIN) console_out << "None found\n";
    else console_out << global_max_path << "\n";
    console_out.close();

    pthread_barrier_destroy(&barrier);
    delete[] handles;
    delete[] dist;
    delete[] changed;
    freeECLgraph(g);
    return 0;
}