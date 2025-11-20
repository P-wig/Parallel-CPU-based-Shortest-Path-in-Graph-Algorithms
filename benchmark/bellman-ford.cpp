//compile: g++ -std=c++17 -O2 -I. -o benchmark/bellman-ford benchmark/bellman-ford.cpp
//run: ./benchmark/bellman-ford internet.egr sssp_results.txt

/*
Bellman-Ford vs. Dijkstra: Key Differences

- Algorithmic Approach:
  - Dijkstra's algorithm uses a priority queue (min-heap) to always select the node with the smallest tentative distance, expanding outward from the source.
  - Bellman-Ford iteratively relaxes all edges for all nodes, repeating this process (nodes-1) times, regardless of current distances.

- Edge Relaxation:
  - Dijkstra only relaxes edges from the current minimum-distance node.
  - Bellman-Ford relaxes every edge in the graph during each iteration.

- Negative Weights:
  - Dijkstra's algorithm only works with non-negative edge weights.
  - Bellman-Ford works with negative edge weights and can detect negative-weight cycles.

- Early Termination:
  - Dijkstra terminates when all reachable nodes have been processed.
  - Bellman-Ford can terminate early if no distances change in an iteration, but otherwise always runs (nodes-1) iterations.

- Complexity:
  - Dijkstra: O((nodes + edges) * log(nodes)) with min-heap.
  - Bellman-Ford: O(nodes * edges).

- Output:
  - Both algorithms produce shortest-path distances from a single source, formatted identically in this project.
  - Bellman-Ford additionally warns if a negative-weight cycle is detected.

Summary:
- Bellman-Ford is more general (handles negative weights), but slower.
- Dijkstra is faster for non-negative weights, but cannot handle negative cycles.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <climits>
#include <chrono>
#include <vector>
#include "ECLgraph.h"

// Sequential Bellman-Ford algorithm from a single source
static bool bellman_ford(const ECLgraph& g, int source, int* dist) {
    int n = g.nodes;
    for (int i = 0; i < n; i++) dist[i] = INT_MAX;
    dist[source] = 0;

    // Relax all edges n-1 times
    for (int iter = 1; iter < n; iter++) {
        bool changed = false;
        for (int u = 0; u < n; u++) {
            int start = g.nindex[u];
            int end = g.nindex[u + 1];
            for (int i = start; i < end; i++) {
                int v = g.nlist[i];
                int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    changed = true;
                }
            }
        }
        if (!changed) break; // Early exit if no changes
    }

    // Check for negative-weight cycles
    for (int u = 0; u < n; u++) {
        int start = g.nindex[u];
        int end = g.nindex[u + 1];
        for (int i = start; i < end; i++) {
            int v = g.nlist[i];
            int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                return false; // Negative cycle detected
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "Single-Source Shortest Path using Bellman-Ford\n";

    if (argc != 2 && argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file [output_file]\n";
        exit(-1);
    }

    std::string output_file = (argc == 3) ? std::string("benchmark/") + argv[2] : "benchmark/bellman_ford_results.txt";

    ECLgraph g = readECLgraph(argv[1]);
    std::cout << "input: " << argv[1] << "\n";
    std::cout << "output: " << output_file << "\n";
    std::cout << "nodes: " << g.nodes << "\n";
    std::cout << "edges: " << g.edges << "\n";
    if (g.eweight != NULL) std::cout << "graph has edge weights\n";
    else std::cout << "graph has no edge weights (using weight = 1)\n";

    int* dist = new int[g.nodes];

    auto beg = std::chrono::high_resolution_clock::now();

    int source = 0;
    bool ok = bellman_ford(g, source, dist);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - beg;
    std::cout << "\ncompute time: " << runtime.count() << " s\n";

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
            if (dist[i] > global_max_path) global_max_path = dist[i];
        }
    }
    outfile.close();

    if (!ok) std::cout << "WARNING: Negative-weight cycle detected!\n";
    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Global max shortest-path: ";
    if (global_max_path == INT_MIN) std::cout << "None found\n";
    else std::cout << global_max_path << "\n";

    delete[] dist;
    freeECLgraph(g);
    return 0;
}