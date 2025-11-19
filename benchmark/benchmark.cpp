//compile: g++ -std=c++17 -O2 -I. -o benchmark/benchmark benchmark/benchmark.cpp
//run: ./benchmark/benchmark internet.egr sssp_results.txt

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <chrono>
#include <algorithm>
#include <queue>
#include <vector>
#include "ECLgraph.h"

// Sequential Dijkstra's algorithm from a single source
static void dijkstra(const ECLgraph& g, int source, int* dist) {
    // Initialize distances
    for (int i = 0; i < g.nodes; i++) {
        dist[i] = INT_MAX;
    }

    dist[source] = 0;

    // Min-heap: pair of (distance, node)
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> queue;
    queue.push({0, source});

    while (!queue.empty()){
        int curr_dist = queue.top().first;
        int curr_node = queue.top().second;
        queue.pop();

        // Skip if we've already found a better path
        if (curr_dist > dist[curr_node]) continue; 

        // Update distances of adjacent vertices
        const int start = g.nindex[curr_node];
        const int end = g.nindex[curr_node + 1];

        for (int i = start; i < end; i++) {
            int v = g.nlist[i];
            int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
            int new_dist = dist[curr_node] + weight;

            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                queue.push({new_dist, v});
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Single-Source Shortest Path using Sequential Dijkstra (Min-Heap)\n";

    // Check command line
    if (argc != 2 && argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file [output_file]\n";
        exit(-1);
    }

    // Set output file path to always be in benchmark directory
    std::string output_file = (argc == 3) ? std::string("benchmark/") + argv[2] : "benchmark/sssp_results.txt";

    // Read input
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

    // Allocate distance array
    int* dist = new int[g.nodes];

    // Start time
    auto beg = std::chrono::high_resolution_clock::now();

    // Execute timed code - compute shortest paths from node 0
    int source = 0;
    dijkstra(g, source, dist);

    // End time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate runtime
    std::chrono::duration<double> runtime = end - beg;
    std::cout << "\ncompute time: " << runtime.count() << " s\n";

    // Write results to file and find global max shortest-path
    std::ofstream outfile(output_file);
    if (!outfile) {
        std::cerr << "ERROR: could not open output file\n";
        exit(-1);
    }

    outfile << "# Single-Source Shortest Path from node " << source << "\n";
    outfile << "# Format: target_node distance\n";

    int global_max_path = INT_MIN;
    for (int i = 0; i < g.nodes; i++) {
        if (dist[i] == INT_MAX) {
            outfile << i << " INF\n";
        } else {
            outfile << i << " " << dist[i] << "\n";
            if (dist[i] > global_max_path){
                global_max_path = dist[i];
            }
        }
    }

    outfile.close();

    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Global max shortest-path: ";
    if (global_max_path == INT_MIN) std::cout << "None found\n";
    else std::cout << global_max_path << "\n";

    // Clean up
    delete[] dist;
    freeECLgraph(g);
    return 0;
}