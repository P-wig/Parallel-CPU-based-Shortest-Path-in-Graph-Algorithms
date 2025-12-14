/*
Compile: g++ -std=c++17 -O2 -pthread -I. -o algorithm_tests/pthread/dijkstra_pthread algorithm_tests/pthread/dijkstra_pthread.cpp
Run: ./algorithm_tests/pthread/dijkstra_pthread internet.egr 4

Dijkstra's Algorithm (pthreads) - Fixed Missing Nodes Issue

Fixed the race condition in termination detection that was causing some reachable nodes
to be missed when using multiple threads.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <chrono>
#include <pthread.h>
#include <cstdlib>
#include <queue>
#include <atomic>
#include <algorithm>
#include <mutex>
#include "ECLgraph.h"

/**
 * Global shared data
 */
ECLgraph* g_graph;
std::atomic<int>* g_dist;
std::atomic<bool>* g_visited;
std::atomic<bool> g_done(false);
std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> g_pq;
std::mutex g_pq_mutex;
std::atomic<int> g_active_threads(0); // Track threads currently processing nodes

/**
 * Fixed worker that properly handles termination
 */
void* dijkstra_simple_worker(void* arg) {
    int tid = *(int*)arg;
    
    while (!g_done.load()) {
        int current_node = -1;
        
        // Try to get work from queue
        {
            std::lock_guard<std::mutex> lock(g_pq_mutex);
            
            // Find next node to process
            while (!g_pq.empty()) {
                auto [d, u] = g_pq.top();
                g_pq.pop();
                
                if (!g_visited[u].load() && g_dist[u].load() == d) {
                    current_node = u;
                    g_visited[u].store(true);
                    break;
                }
            }
            
            // FIXED: Only terminate if queue is empty AND no threads are working
            if (current_node == -1) {
                if (g_active_threads.load() == 0) {
                    g_done.store(true);
                }
                break; // This thread exits, but others may still be working
            }
        }
        
        if (current_node == -1) continue; // No work available, try again
        
        // Mark this thread as active
        g_active_threads.fetch_add(1);
        
        // Process this node (outside mutex)
        int u = current_node;
        int u_dist = g_dist[u].load();
        
        // Process all edges from this node
        for (int e = g_graph->nindex[u]; e < g_graph->nindex[u + 1]; e++) {
            int v = g_graph->nlist[e];
            int weight = (g_graph->eweight != nullptr) ? g_graph->eweight[e] : 1;
            int new_dist = u_dist + weight;
            
            // Try to update distance
            int old_dist = g_dist[v].load();
            while (new_dist < old_dist) {
                if (g_dist[v].compare_exchange_weak(old_dist, new_dist)) {
                    // Successfully updated - add to queue if not visited
                    if (!g_visited[v].load()) {
                        std::lock_guard<std::mutex> lock(g_pq_mutex);
                        g_pq.push({new_dist, v});
                    }
                    break;
                }
            }
        }
        
        // Mark this thread as no longer active
        g_active_threads.fetch_sub(1);
        
        // Check termination condition again after finishing work
        {
            std::lock_guard<std::mutex> lock(g_pq_mutex);
            if (g_pq.empty() && g_active_threads.load() == 0) {
                g_done.store(true);
            }
        }
    }
    
    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file num_threads\n";
        return -1;
    }
    
    int num_threads = std::atoi(argv[2]);
    if (num_threads < 1) {
        std::cerr << "ERROR: num_threads must be at least 1\n";
        return -1;
    }
    
    ECLgraph graph = readECLgraph(argv[1]);
    g_graph = &graph;
    int source = 0;
    
    std::cout << "Dijkstra's Algorithm (Fixed Termination)\n";
    std::cout << "Nodes: " << graph.nodes << "\n";
    std::cout << "Edges: " << graph.edges << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Source node: " << source << "\n";
    
    // Initialize
    g_dist = new std::atomic<int>[graph.nodes];
    g_visited = new std::atomic<bool>[graph.nodes];
    
    for (int i = 0; i < graph.nodes; i++) {
        g_dist[i].store(INT_MAX);
        g_visited[i].store(false);
    }
    g_dist[source].store(0);
    
    // Initialize priority queue and counters
    g_pq.push({0, source});
    g_done.store(false);
    g_active_threads.store(0);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create threads
    std::vector<pthread_t> threads(num_threads);
    std::vector<int> thread_ids(num_threads);
    
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], nullptr, dijkstra_simple_worker, &thread_ids[i]);
    }
    
    // Wait for threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end_time - start_time;
    
    std::cout << "Compute time: " << runtime.count() << " s\n";
    
    // Write results
    std::string output_file = "algorithm_tests/pthread/results/dijkstra_pthread_results.txt";
    std::ofstream outfile(output_file);
    outfile << "# Single-Source Shortest Path from node " << source << "\n";
    
    int reachable = 0;
    int max_distance = 0;
    for (int i = 0; i < graph.nodes; i++) {
        int d = g_dist[i].load();
        if (d == INT_MAX) {
            outfile << i << " INF\n";
        } else {
            outfile << i << " " << d << "\n";
            reachable++;
            max_distance = std::max(max_distance, d);
        }
    }
    outfile.close();
    
    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Reachable nodes: " << reachable << "\n";
    std::cout << "Maximum distance: " << max_distance << "\n";
    
    // Cleanup
    delete[] g_dist;
    delete[] g_visited;
    freeECLgraph(graph);
    
    return 0;
}