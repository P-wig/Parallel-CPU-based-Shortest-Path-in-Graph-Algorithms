/*
Compile: mpic++ -std=c++17 -O2 -I. -o algorithm_tests/mpi/bellman-ford_mpi algorithm_tests/mpi/bellman-ford_mpi.cpp
Run: mpiexec -n 4 ./algorithm_tests/mpi/bellman-ford_mpi internet.egr sssp_results.txt
*/

/*
Why Bellman-Ford Works Well with MPI (Distributed Parallelism)

- Bulk Synchronous Computation:
  - Bellman-Ford relaxes all edges in the graph for every node, in every iteration.
  - This "bulk" edge relaxation is naturally parallelizable: each MPI rank can process a subset of nodes or edges independently.

- Simple Communication Pattern:
  - After each iteration, all ranks synchronize their distance arrays (e.g., with MPI_Allreduce).
  - Communication is infrequent (once per iteration), and the data exchanged is compact (just the distance array).

- No Fine-Grained Dependencies:
  - Unlike algorithms that require immediate propagation of updates (e.g., Delta-Stepping), Bellman-Ford only needs global updates after each full relaxation.
  - This reduces communication overhead and avoids idle time due to waiting for other ranks.

- Good Load Balancing:
  - The work (relaxing edges) can be evenly distributed among ranks, minimizing load imbalance.

- Scalability:
  - As the number of ranks increases, the edge relaxation workload is divided, leading to a recordable speedup.
  - MPI overhead is low because communication is batched and infrequent.

Summary:
Bellman-Ford's structure—bulk edge relaxation and simple, infrequent synchronization—makes it well-suited for distributed-memory parallelism with MPI. This leads to efficient scaling and observable speedup compared to the sequential version.
*/

#include <iostream>
#include <fstream>
#include <climits>
#include <vector>
#include <chrono>
#include <mpi.h>
#include "ECLgraph.h"

// Parallel Bellman-Ford using MPI
bool bellman_ford_mpi(const ECLgraph& g, int source, std::vector<int>& dist, int rank, int size) {
    int n = g.nodes;
    dist.assign(n, INT_MAX);
    dist[source] = 0;

    // Block partitioning: each rank processes a subset of nodes
    int block_size = (n + size - 1) / size;
    int block_start = rank * block_size;
    int block_end = std::min(n, block_start + block_size);

    for (int iter = 1; iter < n; iter++) {
        bool changed_local = false;
        // Block partitioning: each rank processes a contiguous block of nodes
        for (int u = block_start; u < block_end; u++) {
            int start = g.nindex[u];
            int end = g.nindex[u + 1];
            for (int i = start; i < end; i++) {
                int v = g.nlist[i];
                int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
                if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    changed_local = true;
                }
            }
        }
        // Synchronize distances across all ranks
        MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        int changed_global = changed_local ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &changed_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (!changed_global) break;
    }
    return true;
    // Negative cycle check (optional, only rank 0 reports)
    /*
    bool negative_cycle = false;
    for (int u = rank; u < n; u += size) {
        int start = g.nindex[u];
        int end = g.nindex[u + 1];
        for (int i = start; i < end; i++) {
            int v = g.nlist[i];
            int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                negative_cycle = true;
            }
        }
    }
    int negative_cycle_global = negative_cycle ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &negative_cycle_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return !negative_cycle_global;
    */
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
        : "algorithm_tests/mpi/results/bellman_ford_mpi_results.txt";

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

    if (rank == 0) {
        std::cout << "Single-Source Shortest Path using Bellman-Ford with MPI\n";
        std::cout << "input: " << argv[1] << "\n";
        std::cout << "output: " << output_file << "\n";
        std::cout << "nodes: " << g.nodes << "\n";
        std::cout << "edges: " << g.edges << "\n";
        if (g.eweight != nullptr) std::cout << "graph has edge weights\n";
        else std::cout << "graph has no edge weights (using weight = 1)\n";
        std::cout << "MPI ranks used: " << size << "\n";
    }

    std::vector<int> dist(g.nodes, INT_MAX);

    MPI_Barrier(MPI_COMM_WORLD);
    auto beg = std::chrono::high_resolution_clock::now();

    int source = 0;
    bool ok = bellman_ford_mpi(g, source, dist, rank, size);

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
        if (!ok) std::cout << "WARNING: Negative-weight cycle detected!\n";
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