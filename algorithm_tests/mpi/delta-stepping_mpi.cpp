/*
Compile: mpic++ -std=c++17 -O2 -I. -o algorithm_tests/mpi/delta-stepping_mpi algorithm_tests/mpi/delta-stepping_mpi.cpp
Run: mpiexec -n 4 ./algorithm_tests/mpi/delta-stepping_mpi internet.egr sssp_results.txt
*/

/*
Delta-Stepping MPI Implementation: Deficits and Performance Analysis

Deficits of This MPI Program:
- High Communication Overhead: The algorithm requires frequent global synchronization (MPI_Allreduce, MPI_Allgather) after every bucket and relaxation step. This incurs significant latency, especially as the number of nodes (MPI ranks) increases.
- Fine-Grained Dependencies: Delta-Stepping relies on rapid propagation of distance updates, which is efficient in shared memory but slow in distributed memory due to message passing.
- Poor Scalability: Each rank must synchronize the entire distance array and bucket contents, leading to large data transfers and idle time while waiting for other ranks.
- Load Imbalance: The distribution of nodes and edges may be uneven, causing some ranks to finish their work much earlier than others, further increasing idle time.

Why 1 Node Is Fast, but 2 or 4 Nodes Are Slow:
- With 1 node, all computation is local and there is no MPI communication, so the algorithm runs at full speed (0.1843 seconds).
- With 2 or 4 nodes, every bucket and relaxation step requires global communication, which dominates runtime (30 seconds), even though the computation per rank is reduced.
- The cost of synchronizing large arrays and coordinating updates across ranks far outweighs the benefits of parallel computation for this algorithm and graph size.

Summary:
Delta-Stepping is designed for shared-memory parallelism (threads, OpenMP), not distributed-memory (MPI). The frequent, fine-grained communication required by the algorithm makes MPI implementations slow and poorly scalable for most real-world graphs.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <climits>
#include <chrono>
#include <mpi.h>
#include "ECLgraph.h"
#include <algorithm>
#include <cmath>

// Parallel Delta-Stepping SSSP (MPI, block partitioning)
void delta_stepping_mpi(const ECLgraph& g, int source, std::vector<int>& dist, int delta, int rank, int size) {
    int n = g.nodes;
    dist.assign(n, INT_MAX);
    dist[source] = 0;

    std::vector<std::set<int>> buckets((n * delta) / delta + 2);
    if (source % size == rank)
        buckets[0].insert(source);

    int current_bucket = 0;
    while (current_bucket < buckets.size()) {
        std::set<int> S;
        // Light edge relaxation
        while (true) {
            // Gather all nodes in current bucket across ranks
            std::vector<int> local_nodes(buckets[current_bucket].begin(), buckets[current_bucket].end());
            int local_count = local_nodes.size();
            std::vector<int> all_nodes;
            int total_count = 0;
            std::vector<int> counts(size);
            MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            for (int c : counts) total_count += c;
            std::vector<int> displs(size);
            displs[0] = 0;
            for (int i = 1; i < size; i++) displs[i] = displs[i-1] + counts[i-1];
            all_nodes.resize(total_count);
            MPI_Allgatherv(local_nodes.data(), local_count, MPI_INT,
                           all_nodes.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

            if (all_nodes.empty()) break;
            S.insert(all_nodes.begin(), all_nodes.end());
            buckets[current_bucket].clear();

            // Light edge relaxation (round-robin)
            for (int u : S) {
                if (u % size != rank) continue;
                int start = g.nindex[u];
                int end = g.nindex[u + 1];
                for (int i = start; i < end; i++) {
                    int v = g.nlist[i];
                    int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                    if (weight <= delta) {
                        int new_dist = dist[u] + weight;
                        if (new_dist < dist[v]) {
                            dist[v] = new_dist;
                            int b = new_dist / delta;
                            buckets[b].insert(v);
                        }
                    }
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        }
        // Heavy edge relaxation (round-robin)
        for (int u : S) {
            if (u % size != rank) continue;
            int start = g.nindex[u];
            int end = g.nindex[u + 1];
            for (int i = start; i < end; i++) {
                int v = g.nlist[i];
                int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                if (weight > delta) {
                    int new_dist = dist[u] + weight;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        int b = new_dist / delta;
                        buckets[b].insert(v);
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        current_bucket++;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2 && argc != 3) {
        if (rank == 0)
            std::cerr << "USAGE: " << argv[0] << " input_file [output_file]\n";
        MPI_Finalize();
        return -1;
    }

    std::string output_file = (argc == 3)
        ? std::string("algorithm_tests/mpi/results/") + argv[2]
        : "algorithm_tests/mpi/results/delta_stepping_mpi_results.txt";

    // Only rank 0 reads the graph
    ECLgraph g;
    if (rank == 0) {
        g = readECLgraph(argv[1]);
    }
    // Broadcast graph sizes
    MPI_Bcast(&g.nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate arrays on other ranks
    if (rank != 0) {
        g.nindex = new int[g.nodes + 1];
        g.nlist = new int[g.edges];
        g.eweight = nullptr;
    }
    MPI_Bcast(g.nindex, g.nodes + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.nlist, g.edges, MPI_INT, 0, MPI_COMM_WORLD);
    int has_eweight = (g.eweight != nullptr) ? 1 : 0;
    MPI_Bcast(&has_eweight, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (has_eweight) {
        if (rank != 0) g.eweight = new int[g.edges];
        MPI_Bcast(g.eweight, g.edges, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        g.eweight = nullptr;
    }

    // Calculate average edge weight (all ranks get the same value)
    int average_edge_weight = 1;
    if (g.eweight != nullptr && g.edges > 0) {
        long long sum_weights = 0;
        if (rank == 0) {
            for (int i = 0; i < g.edges; i++) {
                sum_weights += g.eweight[i];
            }
        }
        // Broadcast sum_weights from rank 0
        MPI_Bcast(&sum_weights, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        average_edge_weight = static_cast<int>(sum_weights / g.edges);
    }

    // Set delta to average edge weight
    int delta = std::max(1, average_edge_weight);

    if (rank == 0) {
        std::cout << "input: " << argv[1] << "\n";
        std::cout << "output: " << output_file << "\n";
        std::cout << "nodes: " << g.nodes << "\n";
        std::cout << "edges: " << g.edges << "\n";
        if (g.eweight != nullptr) std::cout << "graph has edge weights\n";
        else std::cout << "graph has no edge weights (using weight = 1)\n";
        std::cout << "delta (bucket width) chosen: " << delta << "\n";
        std::cout << "MPI ranks used: " << size << "\n";
    }

    std::vector<int> dist(g.nodes, INT_MAX);

    MPI_Barrier(MPI_COMM_WORLD);
    auto beg = std::chrono::high_resolution_clock::now();

    int source = 0;
    delta_stepping_mpi(g, source, dist, delta, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - beg;

    // Only rank 0 writes results
    if (rank == 0) {
        std::ofstream outfile(output_file);
        if (!outfile) {
            std::cerr << "ERROR: could not open output file\n";
            MPI_Finalize();
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
        std::cout << "\ncompute time: " << runtime.count() << " s\n";
        std::cout << "Results written to " << output_file << "\n";
        std::cout << "Global max shortest-path: ";
        if (global_max_path == INT_MIN) std::cout << "None found\n";
        else std::cout << global_max_path << "\n";
    }

    freeECLgraph(g);
    MPI_Finalize();
    return 0;
}