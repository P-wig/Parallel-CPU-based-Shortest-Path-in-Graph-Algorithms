/*
Compile: g++ -std=c++17 -O2 -pthread -I. -o algorithm_tests/pthread/dijkstra_pthread algorithm_tests/pthread/dijkstra_pthread.cpp
Run: ./algorithm_tests/pthread/dijkstra_pthread internet.egr 4

Dijkstra's Algorithm: Parallelization Deficits (pthreads/shared-memory)

- Inherently Sequential: Dijkstra's algorithm relies on selecting the global minimum unvisited node in each iteration, which is a fundamentally sequential operation. This limits parallelism to only the edge relaxation phase for the current node.

- Avoiding the Priority Queue: Instead of using a priority queue, thread 0 performs a linear scan of the dist[] array to find the global minimum unvisited node in each iteration. This is possible because all threads share memory, and the scan is efficient for moderate graph sizes. This approach avoids the complexity and synchronization overhead of a concurrent priority queue.

- Poor Scalability: For most real-world graphs, the number of neighbors per node is small compared to the number of nodes. As a result, most threads are idle during edge relaxation, and increasing thread count leads to diminishing returns or even slower performance.

- High Synchronization Overhead: Every iteration requires all threads to synchronize at barriers. With more threads, the cost of synchronization increases, often outweighing the benefits of parallel edge relaxation.

- Work Imbalance: The workload per iteration is highly variable and often too small to keep all threads busy, resulting in poor load balancing.

- Memory Contention: Multiple threads updating shared data structures (e.g., dist[]) can lead to cache contention and false sharing, further reducing performance.

- Not Suitable for Negative Weights: Dijkstra's algorithm cannot handle graphs with negative edge weights.

Summary: 
Parallel Dijkstra with pthreads is correct but not efficient for large graphs. For better scalability, use algorithms like Bellman-Ford or Delta-Stepping, which allow more parallelism and better load balancing.
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
static bool* visited;
static int source = 0;
static int curr_node;
static int curr_dist;
static int start_e, end_e;
static bool done = false;

static void* dijkstra_pthread(void* arg) {
    const int tid = (int)(intptr_t)arg;

    while (true) {
        // Thread 0 selects the global minimum unvisited node
        if (tid == 0) {
            curr_node = -1;
            curr_dist = INT_MAX;
            for (int u = 0; u < g.nodes; u++) {
                if (!visited[u] && dist[u] < curr_dist) {
                    curr_dist = dist[u];
                    curr_node = u;
                }
            }
            if (curr_node == -1) done = true;
            else {
                visited[curr_node] = true;
                start_e = g.nindex[curr_node];
                end_e = g.nindex[curr_node + 1];
            }
        }

        pthread_barrier_wait(&barrier);

        if (done) break;

        // All threads relax a block of neighbors
        int total_edges = end_e - start_e;
        int block = (total_edges + threads - 1) / threads;
        int begin = start_e + tid * block;
        int finish = std::min(end_e, begin + block);
        for (int i = begin; i < finish; i++) {
            int v = g.nlist[i];
            int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
            int new_dist = curr_dist + weight;
            if (new_dist < dist[v]) {
                dist[v] = new_dist;
            }
        }

        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    std::cout << "Single-Source Shortest Path using Dijkstra with pthreads (clean C-style)\n";

    if (argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file num_threads\n";
        return -1;
    }

    threads = std::atoi(argv[2]);
    if (threads < 1) {
        std::cerr << "ERROR: num_threads must be at least 1\n";
        return -1;
    }

    std::string output_file = "algorithm_tests/pthread/results/dijkstra_pthread_results.txt";

    g = readECLgraph(argv[1]);
    std::cout << "input: " << argv[1] << "\n";
    std::cout << "output: " << output_file << "\n";
    std::cout << "nodes: " << g.nodes << "\n";
    std::cout << "edges: " << g.edges << "\n";
    if (g.eweight != NULL) std::cout << "graph has edge weights\n";
    else std::cout << "graph has no edge weights (using weight = 1)\n";
    std::cout << "pthreads used: " << threads << "\n";

    dist = new int[g.nodes];
    visited = new bool[g.nodes];
    for (int i = 0; i < g.nodes; i++) {
        dist[i] = INT_MAX;
        visited[i] = false;
    }
    dist[source] = 0;

    pthread_barrier_init(&barrier, nullptr, threads);
    pthread_t* handles = new pthread_t[threads - 1];

    auto beg = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < threads - 1; t++) {
        pthread_create(&handles[t], nullptr, dijkstra_pthread, (void*)(intptr_t)t);
    }

    // Main thread also participates
    dijkstra_pthread((void*)(intptr_t)(threads - 1));

    for (int t = 0; t < threads - 1; t++) {
        pthread_join(handles[t], nullptr);
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

    int reachable = 0;
    for (int i = 0; i < g.nodes; i++) {
        if (dist[i] != INT_MAX) reachable++;
    }
    std::cout << "Reachable nodes from source: " << reachable << "\n";

    pthread_barrier_destroy(&barrier);
    delete[] handles;
    delete[] dist;
    delete[] visited;
    freeECLgraph(g);
    return 0;
}