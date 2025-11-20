//compile: g++ -std=c++17 -O2 -I. -o benchmark/delta-stepping benchmark/delta-stepping.cpp
//run: ./benchmark/delta-stepping internet.egr sssp_results.txt

/*
Delta-Stepping Algorithm: Summary and Comparison to Dijkstra

How Delta-Stepping Works:
- Delta-stepping is a single-source shortest path (SSSP) algorithm designed for graphs with non-negative edge weights.
- The algorithm divides the search into "buckets" based on tentative distances, where each bucket contains nodes with distances in [b*delta, (b+1)*delta).
- It processes nodes bucket by bucket, relaxing "light" edges (weight <= delta) first, then "heavy" edges (weight > delta).
- Light edge relaxation is performed repeatedly within a bucket until no new nodes are added; heavy edges are relaxed once per bucket.
- This bucketed approach allows for more parallelism and batch processing, making delta-stepping suitable for parallel and distributed systems, but it can also be used sequentially.

Key Differences from Dijkstra:
- Dijkstra's algorithm uses a global priority queue (min-heap) to always select the node with the smallest tentative distance, expanding outward in strict order.
- Delta-stepping relaxes nodes in batches (buckets), allowing multiple nodes with similar distances to be processed together, which can reduce the number of priority queue operations and increase parallelism.
- Dijkstra is strictly sequential in its node selection; delta-stepping is more flexible and can be parallelized more easily.
- Delta-stepping separates edge relaxations into "light" and "heavy" edges, while Dijkstra treats all edges the same.
- Both algorithms produce correct shortest-path distances for non-negative weights, but delta-stepping can be tuned (via the delta parameter) for performance and parallel scalability.

Summary:
- Delta-stepping is a bucket-based SSSP algorithm that enables efficient batch processing and parallelization.
- It differs from Dijkstra by relaxing nodes in groups and separating edge types, making it more suitable for parallel and distributed environments.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <climits>
#include <chrono>
#include <vector>
#include <queue>
#include <set>
#include "ECLgraph.h"

// Sequential Delta-Stepping SSSP (for non-negative weights)
static void delta_stepping(const ECLgraph& g, int source, int* dist, int delta) {
    int n = g.nodes;
    for (int i = 0; i < n; i++) dist[i] = INT_MAX;
    dist[source] = 0;

    // Buckets: each bucket contains nodes with tentative distances in [b*delta, (b+1)*delta)
    std::vector<std::set<int>> buckets((n * delta) / delta + 2); // oversize for safety
    buckets[0].insert(source);

    int current_bucket = 0;
    while (current_bucket < buckets.size()) {
        // Set of nodes to process in this bucket
        std::set<int> S;
        // Light edge relaxation
        while (!buckets[current_bucket].empty()) {
            S.insert(buckets[current_bucket].begin(), buckets[current_bucket].end());
            buckets[current_bucket].clear();

            for (int u : S) {
                int start = g.nindex[u];
                int end = g.nindex[u + 1];
                for (int i = start; i < end; i++) {
                    int v = g.nlist[i];
                    int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                    if (weight <= delta) { // light edge
                        int new_dist = dist[u] + weight;
                        if (new_dist < dist[v]) {
                            dist[v] = new_dist;
                            int b = new_dist / delta;
                            buckets[b].insert(v);
                        }
                    }
                }
            }
        }
        // Heavy edge relaxation
        for (int u : S) {
            int start = g.nindex[u];
            int end = g.nindex[u + 1];
            for (int i = start; i < end; i++) {
                int v = g.nlist[i];
                int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                if (weight > delta) { // heavy edge
                    int new_dist = dist[u] + weight;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        int b = new_dist / delta;
                        buckets[b].insert(v);
                    }
                }
            }
        }
        current_bucket++;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Single-Source Shortest Path using Sequential Delta-Stepping\n";

    if (argc != 2 && argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file [output_file]\n";
        exit(-1);
    }

    std::string output_file = (argc == 3) ? std::string("benchmark/") + argv[2] : "benchmark/delta_stepping_results.txt";

    ECLgraph g = readECLgraph(argv[1]);
    std::cout << "input: " << argv[1] << "\n";
    std::cout << "output: " << output_file << "\n";
    std::cout << "nodes: " << g.nodes << "\n";
    std::cout << "edges: " << g.edges << "\n";
    if (g.eweight != NULL) std::cout << "graph has edge weights\n";
    else std::cout << "graph has no edge weights (using weight = 1)\n";

    // Calculate average edge weight
    int average_edge_weight = 1;
    if (g.eweight != NULL && g.edges > 0) {
        long long sum_weights = 0;
        for (int i = 0; i < g.edges; i++) {
            sum_weights += g.eweight[i];
        }
        average_edge_weight = static_cast<int>(sum_weights / g.edges);
    }

    int* dist = new int[g.nodes];

    auto beg = std::chrono::high_resolution_clock::now();

    int source = 0;
    int delta = std::max(1, average_edge_weight); // Use average edge weight for delta
    std::cout << "delta (bucket width) chosen: " << delta << "\n";
    delta_stepping(g, source, dist, delta);

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

    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Global max shortest-path: ";
    if (global_max_path == INT_MIN) std::cout << "None found\n";
    else std::cout << global_max_path << "\n";

    delete[] dist;
    freeECLgraph(g);
    return 0;
}