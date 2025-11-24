/*
Compile: mpic++ -std=c++17 -O2 -I. -o algorithm_tests/mpi/dijkstra_mpi algorithm_tests/mpi/dijkstra_mpi.cpp
Run: mpiexec -n 4 ./algorithm_tests/mpi/dijkstra_mpi internet.egr
*/

/*
Dijkstra's Algorithm: Results and Drawbacks in Parallel (MPI) Implementation

Results:
- Dijkstra's algorithm computes the shortest path from a single source to all other nodes in a weighted graph with non-negative edge weights.
- In this MPI implementation, rank 0 manages the global priority queue, while all ranks participate in edge relaxation for the current node.
- The algorithm produces correct shortest-path results and can handle large graphs (e.g., 124,651 nodes, 387,240 edges).
- Example output: "Global max shortest-path: 6836" and compute time of ~109 seconds for a large graph.

Drawbacks:
- Dijkstra's algorithm is inherently sequential due to its reliance on a global minimum selection from the priority queue.
- Parallelization with MPI is limited: only edge relaxation is distributed, while queue management and node selection remain serial.
- High communication overhead: frequent broadcasts and reductions are required to synchronize state across ranks.
- Poor scalability: speedup is limited, and performance gains diminish as the number of ranks increases.
- Block partitioning or distributed queues break algorithm correctness, as the next node to process must always be globally minimal.
- For large graphs or high-performance needs, more parallel-friendly algorithms (e.g., Δ-stepping, parallel BFS) are recommended.

Summary:
- MPI Dijkstra is correct but slow for large graphs.
- more worker nodes inversely affects performance due to communication overhead.
- Use other parallel SSSP algorithms for better scalability in distributed environments.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <climits>
#include <chrono>
#include <mpi.h>
#include "ECLgraph.h"

// MPI Dijkstra: rank 0 manages the queue, all ranks relax edges
void mpi_dijkstra(const ECLgraph& g, int source, std::vector<int>& dist, int rank, int size) {
    int n = g.nodes;
    std::vector<bool> visited(n, false);
    std::vector<char> updated(n, 0);

    if (rank == 0) dist[source] = 0;

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> queue;
    if (rank == 0) queue.push({0, source});

    while (true) {
        std::fill(updated.begin(), updated.end(), 0);

        int curr_node = -1, curr_dist = INT_MAX;
        if (rank == 0 && !queue.empty()) {
            curr_dist = queue.top().first;
            curr_node = queue.top().second;
            queue.pop();
        }
        MPI_Bcast(&curr_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&curr_dist, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (curr_node == -1) break;

        if (!visited[curr_node]) {
            visited[curr_node] = true;
            int start = g.nindex[curr_node];
            int end = g.nindex[curr_node + 1];
            int edge_count = end - start;
            int block_size = (edge_count + size - 1) / size;
            int block_start = start + rank * block_size;
            int block_end = std::min(end, block_start + block_size);
            for (int i = block_start; i < block_end; i++) {
                int v = g.nlist[i];
                int weight = (g.eweight != nullptr) ? g.eweight[i] : 1;
                int new_dist = curr_dist + weight;
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    updated[v] = 1;
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, updated.data(), n, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int v = 0; v < n; v++) {
                if (updated[v] && !visited[v]) {
                    queue.push({dist[v], v});
                }
            }
        }
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

    std::string output_file = "algorithm_tests/mpi/results/dijkstra_mpi_results.txt";

    // Only rank 0 reads the graph
    ECLgraph g;
    if (rank == 0) {
        g = readECLgraph(argv[1]);
    }
    // Broadcast number of nodes to all ranks
    MPI_Bcast(&g.nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast number of edges to all ranks
    MPI_Bcast(&g.edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // All ranks allocate graph arrays AFTER receiving sizes
    if (rank != 0) {
        g.nindex = new int[g.nodes + 1];
        g.nlist = new int[g.edges];
        g.eweight = nullptr;
    }
    // Broadcast node index array (CSR row pointers) to all ranks
    MPI_Bcast(g.nindex, g.nodes + 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast node list array (CSR column indices) to all ranks
    MPI_Bcast(g.nlist, g.edges, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast edge weights array to all ranks, if present
    int has_eweight = (g.eweight != nullptr) ? 1 : 0;
    MPI_Bcast(&has_eweight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (has_eweight) {
        if (rank != 0) g.eweight = new int[g.edges];
        MPI_Bcast(g.eweight, g.edges, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        g.eweight = nullptr;
    }

    if (rank == 0) {
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
    mpi_dijkstra(g, source, dist, rank, size);

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