/*
Compile: mpic++ -std=c++17 -O2 -I. -o algorithm_tests/mpi/delta-stepping_mpi algorithm_tests/mpi/delta-stepping_mpi.cpp
Run: mpiexec -n 4 ./algorithm_tests/mpi/delta-stepping_mpi internet.egr
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
#include <atomic>

void delta_stepping_mpi(const ECLgraph& g, int source, std::vector<int>& dist, int delta, int rank, int size) {
    int n = g.nodes;
    dist.assign(n, INT_MAX);
    dist[source] = 0;

    int bucket_count = n / delta + 2;
    std::vector<std::set<int>> buckets(bucket_count);
    if (rank == 0) buckets[0].insert(source);

    int curr_bucket = 0;
    while (curr_bucket < bucket_count) {
        // --- Light edge phase: repeat until current bucket is empty globally ---
        std::set<int> S; // All nodes ever in this bucket (for heavy phase)
        while (true) {
            // Gather all nodes in current bucket across ranks
            std::vector<int> local_nodes(buckets[curr_bucket].begin(), buckets[curr_bucket].end());
            int local_count = local_nodes.size();
            std::vector<int> counts(size), displs(size);
            MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            int total_count = 0;
            for (int c : counts) total_count += c;
            displs[0] = 0;
            for (int i = 1; i < size; i++) displs[i] = displs[i-1] + counts[i-1];
            std::vector<int> global_bucket(total_count);
            MPI_Allgatherv(local_nodes.data(), local_count, MPI_INT,
                        global_bucket.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

            // If bucket is empty globally, break
            if (global_bucket.empty()) break;

            // Add all nodes seen in this bucket to S
            S.insert(global_bucket.begin(), global_bucket.end());

            // Each rank processes a block of nodes in the global bucket
            int total = global_bucket.size();
            int block = (total + size - 1) / size;
            int begin = rank * block;
            int end = std::min(total, begin + block);

            std::vector<int> local_dist_updates(n, INT_MAX);
            for (int i = begin; i < end; ++i) {
                int u = global_bucket[i];
                int start = g.nindex[u];
                int finish = g.nindex[u + 1];
                for (int e = start; e < finish; e++) {
                    int v = g.nlist[e];
                    int w = (g.eweight != NULL) ? g.eweight[e] : 1;
                    if (w <= delta) {
                        int new_dist = dist[u] + w;
                        if (new_dist < local_dist_updates[v]) {
                            local_dist_updates[v] = new_dist;
                        }
                    }
                }
            }

            // Allreduce to update global dist
            for (int i = 0; i < n; i++) {
                if (local_dist_updates[i] < dist[i]) {
                    dist[i] = local_dist_updates[i];
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            // Insert changed nodes into their new buckets (light edge phase)
            buckets[curr_bucket].clear();
            for (int i = 0; i < n; ++i) {
                if (local_dist_updates[i] < INT_MAX && dist[i] == local_dist_updates[i]) {
                    int b = dist[i] / delta;
                    // Allow b == curr_bucket for repeated light relaxations
                    if (b < bucket_count && b >= curr_bucket) {
                        buckets[b].insert(i);
                    }
                }
            }

            // --- Synchronize all buckets after this light edge round ---
            for (int b = 0; b < bucket_count; ++b) {
                std::vector<int> local_nodes(buckets[b].begin(), buckets[b].end());
                int local_count = local_nodes.size();
                std::vector<int> counts(size), displs(size);
                MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
                int total_count = 0;
                for (int c : counts) total_count += c;
                displs[0] = 0;
                for (int i = 1; i < size; i++) displs[i] = displs[i-1] + counts[i-1];
                std::vector<int> global_nodes(total_count);
                MPI_Allgatherv(local_nodes.data(), local_count, MPI_INT,
                            global_nodes.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
                buckets[b].clear();
                for (int u : global_nodes) buckets[b].insert(u);
            }
        }

        // --- Heavy edge phase: process all nodes ever in this bucket (S) ---
        // Gather S across all ranks
        std::vector<int> local_S(S.begin(), S.end());
        int local_count = local_S.size();
        std::vector<int> counts(size), displs(size);
        MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int total_count = 0;
        for (int c : counts) total_count += c;
        displs[0] = 0;
        for (int i = 1; i < size; i++) displs[i] = displs[i-1] + counts[i-1];
        std::vector<int> global_S(total_count);
        MPI_Allgatherv(local_S.data(), local_count, MPI_INT,
                    global_S.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        // Each rank processes a block of nodes in S
        int total = global_S.size();
        int block = (total + size - 1) / size;
        int begin = rank * block;
        int end = std::min(total, begin + block);

        std::vector<int> heavy_local_dist_updates(n, INT_MAX);
        for (int i = begin; i < end; ++i) {
            int u = global_S[i];
            int start = g.nindex[u];
            int finish = g.nindex[u + 1];
            for (int e = start; e < finish; e++) {
                int v = g.nlist[e];
                int w = (g.eweight != NULL) ? g.eweight[e] : 1;
                if (w > delta) {
                    int new_dist = dist[u] + w;
                    if (new_dist < heavy_local_dist_updates[v]) {
                        heavy_local_dist_updates[v] = new_dist;
                    }
                }
            }
        }

        // Allreduce to update global dist after heavy phase
        for (int i = 0; i < n; i++) {
            if (heavy_local_dist_updates[i] < dist[i]) {
                dist[i] = heavy_local_dist_updates[i];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Insert changed nodes into their new buckets (heavy edge phase)
        for (int i = 0; i < n; ++i) {
            if (heavy_local_dist_updates[i] < INT_MAX && dist[i] == heavy_local_dist_updates[i]) {
                int b = dist[i] / delta;
                if (b < bucket_count && b > curr_bucket) {
                    buckets[b].insert(i);
                }
            }
        }

        // --- Synchronize all buckets after heavy edge phase ---
        for (int b = 0; b < bucket_count; ++b) {
            std::vector<int> local_nodes(buckets[b].begin(), buckets[b].end());
            int local_count = local_nodes.size();
            std::vector<int> counts(size), displs(size);
            MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            int total_count = 0;
            for (int c : counts) total_count += c;
            displs[0] = 0;
            for (int i = 1; i < size; i++) displs[i] = displs[i-1] + counts[i-1];
            std::vector<int> global_nodes(total_count);
            MPI_Allgatherv(local_nodes.data(), local_count, MPI_INT,
                        global_nodes.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
            buckets[b].clear();
            for (int u : global_nodes) buckets[b].insert(u);
        }

        // Remove all nodes from the current bucket (done for this round)
        buckets[curr_bucket].clear();
        curr_bucket++;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            std::cerr << "USAGE: " << argv[0] << " input_file\n";
        MPI_Finalize();
        return -1;
    }

    std::string output_file = "algorithm_tests/mpi/results/delta_stepping_mpi_results.txt";

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

    // Set delta to average edge weight
    int delta = 1000;

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