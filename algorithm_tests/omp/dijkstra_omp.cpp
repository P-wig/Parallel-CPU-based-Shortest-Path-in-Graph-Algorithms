/*
Compile: g++ -fopenmp -O3 -I../.. -static -o dijkstra_omp dijkstra_omp.cpp

Run: ./dijkstra_omp internet.egr [output_file]
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <chrono>
#include <queue>
#include <vector>
#include <omp.h>
#include "ECLgraph.h"

static void dijkstra_omp(const ECLgraph& g, int source, int* dist) {
    // initialize distances
    #pragma omp parallel for
    for (int i = 0; i < g.nodes; i++) {
        dist[i] = INT_MAX;
    }
    dist[source] = 0;

    // Priority queue: pair of (distance, node)
    typedef std::pair<int, int> pii;
    std::priority_queue<pii, std::vector<pii>, std::greater<pii>> pq;
    pq.push({0, source});

    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();

        // Skip if we've already found a better path
        if (d > dist[u]) continue;

        // Process neighbors
        const int start = g.nindex[u];
        const int end = g.nindex[u + 1];

        for (int i = start; i < end; i++) {
            int v = g.nlist[i];
            int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
            int new_dist = dist[u] + weight;

            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                pq.push({new_dist, v});
            }
        }
    }
}

int main(int argc, char* argv[]) {
  // check command line
  if (argc != 2 && argc != 3) {
    std::cerr << "USAGE: " << argv[0] << " input_file [output_file]\n";
    exit(-1);
  }

  std::string output_file = (argc == 3)
      ? std::string("results/") + argv[2]
      : "results/dijkstra_omp_results.txt";

  // read input
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

  std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";

  // allocate distance array
  int* dist = new int[g.nodes];

  // start time
  auto beg = std::chrono::high_resolution_clock::now();

  // execute timed code - compute shortest paths from node 0
  int source = 0;
  dijkstra_omp(g, source, dist);

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

  std::cout << "\ncompute time: " << runtime.count() << " s\n";
  std::cout << "Results written to " << output_file << "\n";
  std::cout << "Global max shortest-path: ";
  if (global_max_path == INT_MIN) {
    std::cout << "None found\n";
  } else {
    std::cout << global_max_path << "\n";
  }

  // clean up
  delete[] dist;
  freeECLgraph(g);
  return 0;
}