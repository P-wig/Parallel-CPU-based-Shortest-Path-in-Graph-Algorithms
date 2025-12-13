/*
Compile: g++ -std=c++17 -O2 -pthread -I. -o algorithm_tests/pthread/delta-stepping_pthread algorithm_tests/pthread/delta-stepping_pthread.cpp
Run: ./algorithm_tests/pthread/delta-stepping_pthread internet.egr 4

Delta-Stepping Algorithm (pthreads) - Clean Implementation

The Delta-Stepping algorithm is a parallel shortest path algorithm that:
1. Organizes nodes into "buckets" based on their tentative distance
2. Processes buckets sequentially, but processes nodes within a bucket in parallel
3. Uses a delta parameter to determine bucket granularity
4. Relaxes edges and updates distances, potentially adding nodes to future buckets
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <chrono>
#include <pthread.h>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <queue>
#include "ECLgraph.h"

/**
 * ThreadData structure - Contains all data needed by each worker thread
 * 
 * This structure acts as a parameter pack to pass shared data to each thread.
 * All threads share the same graph, distance array, buckets, and synchronization primitives.
 */
struct ThreadData {
    int thread_id;                              // Unique identifier for this thread (0 to num_threads-1)
    int num_threads;                            // Total number of threads in the computation
    ECLgraph* graph;                            // Pointer to the input graph structure
    std::atomic<int>* dist;                     // Atomic distance array - thread-safe shortest distances
    std::vector<std::queue<int>>* buckets;      // Buckets containing nodes to process, organized by distance/delta
    std::vector<pthread_mutex_t>* bucket_mutexes; // One mutex per bucket to protect concurrent access
    pthread_barrier_t* barrier;                // Synchronization barrier for coordinating bucket phases
    std::atomic<int>* current_bucket;           // Index of the bucket currently being processed
    std::atomic<bool>* done;                    // Flag indicating when algorithm should terminate
    std::atomic<int>* active_threads;           // Counter of threads currently processing work
    int delta;                                  // Delta parameter - determines bucket granularity
    int bucket_count;                           // Total number of buckets available
};

/**
 * delta_stepping_worker - Main worker thread function
 * 
 * This is the core parallel algorithm implementation. Each thread:
 * 1. Repeatedly extracts work from the current bucket
 * 2. Processes edges for extracted nodes (relaxation step)
 * 3. Updates distances and adds nodes to appropriate buckets
 * 4. Synchronizes with other threads to advance to next bucket
 * 
 * The algorithm ensures correctness by processing buckets sequentially but
 * allows parallel processing within each bucket.
 */
void* delta_stepping_worker(void* arg) {
    // Extract thread-specific data from the parameter
    ThreadData* data = (ThreadData*)arg;
    int tid = data->thread_id;
    int num_threads = data->num_threads;
    ECLgraph* g = data->graph;
    std::atomic<int>* dist = data->dist;
    auto& buckets = *data->buckets;
    auto& bucket_mutexes = *data->bucket_mutexes;
    pthread_barrier_t* barrier = data->barrier;
    std::atomic<int>* current_bucket = data->current_bucket;
    std::atomic<bool>* done = data->done;
    std::atomic<int>* active_threads = data->active_threads;
    int delta = data->delta;
    int bucket_count = data->bucket_count;
    
    // Main algorithm loop - continues until global termination
    while (!done->load()) {
        // PHASE 1: Process current bucket completely
        // Extract and process nodes from current bucket until empty
        while (true) {
            int curr_b = current_bucket->load();
            if (curr_b >= bucket_count) {
                break; // No more buckets to process
            }
            
            // Extract work from current bucket (thread-safe)
            std::vector<int> local_work;
            pthread_mutex_lock(&bucket_mutexes[curr_b]);
            
            // Take small chunks (3 nodes) to balance load and reduce contention
            // Smaller chunks = better load balancing, less time holding mutex
            for (int i = 0; i < 3 && !buckets[curr_b].empty(); i++) {
                local_work.push_back(buckets[curr_b].front());
                buckets[curr_b].pop();
            }
            
            pthread_mutex_unlock(&bucket_mutexes[curr_b]);
            
            if (local_work.empty()) {
                break; // No more work in current bucket for this thread
            }
            
            // Track that this thread is actively working
            active_threads->fetch_add(1);
            
            // CORE COMPUTATION: Process each extracted node
            for (int u : local_work) {
                int u_dist = dist[u].load(); // Current shortest distance to node u
                
                // Examine all outgoing edges from node u
                for (int e = g->nindex[u]; e < g->nindex[u + 1]; e++) {
                    int v = g->nlist[e];      // Target node of this edge
                    int weight = (g->eweight != nullptr) ? g->eweight[e] : 1; // Edge weight
                    int new_dist = u_dist + weight; // Potential new distance to v
                    
                    // EDGE RELAXATION: Try to improve distance to v
                    int old_dist = dist[v].load();
                    while (new_dist < old_dist) {
                        // Atomic compare-and-swap to update distance
                        if (dist[v].compare_exchange_weak(old_dist, new_dist)) {
                            // Successfully updated distance - add v to appropriate bucket
                            int bucket_id = std::min(new_dist / delta, bucket_count - 1);
                            
                            pthread_mutex_lock(&bucket_mutexes[bucket_id]);
                            buckets[bucket_id].push(v);
                            pthread_mutex_unlock(&bucket_mutexes[bucket_id]);
                            break;
                        }
                        // If CAS failed, old_dist was updated by another thread, retry if still beneficial
                    }
                }
            }
            
            // Finished processing this chunk of work
            active_threads->fetch_sub(1);
        }
        
        // PHASE 2: Synchronization and bucket advancement
        // Wait for all threads to finish processing current bucket
        pthread_barrier_wait(barrier);
        
        // Only thread 0 makes bucket advancement decisions to avoid conflicts
        if (tid == 0) {
            int curr_b = current_bucket->load();
            
            // Check if current bucket received new work during processing
            pthread_mutex_lock(&bucket_mutexes[curr_b]);
            bool current_has_work = !buckets[curr_b].empty();
            pthread_mutex_unlock(&bucket_mutexes[curr_b]);
            
            if (current_has_work) {
                // Stay on current bucket - it has more work to process
                // This implements the "repeat until bucket is empty" requirement
            } else {
                // Current bucket is empty, find next bucket with work
                int next_bucket = curr_b + 1;
                while (next_bucket < bucket_count) {
                    pthread_mutex_lock(&bucket_mutexes[next_bucket]);
                    bool has_work = !buckets[next_bucket].empty();
                    pthread_mutex_unlock(&bucket_mutexes[next_bucket]);
                    
                    if (has_work) break;
                    next_bucket++;
                }
                
                if (next_bucket >= bucket_count) {
                    // No more buckets with work - check for algorithm termination
                    if (active_threads->load() == 0) {
                        done->store(true); // Signal all threads to terminate
                    }
                } else {
                    current_bucket->store(next_bucket); // Advance to next bucket
                }
            }
        }
        
        // Synchronize bucket advancement decision with all threads
        pthread_barrier_wait(barrier);
    }
    
    return nullptr;
}

/**
 * main - Program entry point and algorithm orchestration
 * 
 * Responsibilities:
 * 1. Parse command line arguments
 * 2. Read input graph
 * 3. Initialize data structures (distance array, buckets, synchronization)
 * 4. Create and manage worker threads
 * 5. Measure execution time
 * 6. Write results to output file
 * 7. Clean up resources
 */
int main(int argc, char* argv[]) {
    // Command line argument validation
    if (argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file num_threads\n";
        return -1;
    }
    
    int num_threads = std::atoi(argv[2]);
    if (num_threads < 1) {
        std::cerr << "ERROR: num_threads must be at least 1\n";
        return -1;
    }
    
    // Read input graph and set algorithm parameters
    ECLgraph graph = readECLgraph(argv[1]);
    int delta = 50;                              // Bucket granularity - determines parallelism vs overhead trade-off
    int bucket_count = (7000 / delta) + 10;     // Enough buckets to handle expected maximum distance
    int source = 0;                              // Source node for shortest path computation
    
    // Display configuration information
    std::cout << "Delta-Stepping with pthreads (Clean)\n";
    std::cout << "Nodes: " << graph.nodes << "\n";
    std::cout << "Edges: " << graph.edges << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Delta: " << delta << "\n";
    
    // Initialize distance array - atomic for thread-safe updates
    std::atomic<int>* dist = new std::atomic<int>[graph.nodes];
    for (int i = 0; i < graph.nodes; i++) {
        dist[i].store(INT_MAX); // Initialize all distances to infinity
    }
    dist[source].store(0); // Source node has distance 0
    
    // Initialize bucket data structures
    std::vector<std::queue<int>> buckets(bucket_count);           // One queue per bucket
    std::vector<pthread_mutex_t> bucket_mutexes(bucket_count);    // One mutex per bucket for thread safety
    
    for (int i = 0; i < bucket_count; i++) {
        pthread_mutex_init(&bucket_mutexes[i], nullptr);
    }
    
    buckets[0].push(source); // Start algorithm with source node in bucket 0
    
    // Initialize synchronization primitives
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, num_threads); // Barrier for all threads
    std::atomic<int> current_bucket(0);      // Start processing from bucket 0
    std::atomic<bool> done(false);           // Algorithm termination flag
    std::atomic<int> active_threads(0);      // Count of threads currently processing work
    
    // Create thread data structures - each thread gets same shared data
    std::vector<ThreadData> thread_data(num_threads);
    for (int i = 0; i < num_threads; i++) {
        thread_data[i] = {i, num_threads, &graph, dist, &buckets, &bucket_mutexes,
                         &barrier, &current_bucket, &done, &active_threads, delta, bucket_count};
    }
    
    // Start timing the core algorithm
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create worker threads (n-1 threads, main thread will also work)
    std::vector<pthread_t> threads(num_threads - 1);
    for (int i = 0; i < num_threads - 1; i++) {
        pthread_create(&threads[i], nullptr, delta_stepping_worker, &thread_data[i]);
    }
    
    // Main thread also participates in computation
    delta_stepping_worker(&thread_data[num_threads - 1]);
    
    // Wait for all worker threads to complete
    for (int i = 0; i < num_threads - 1; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;
    
    std::cout << "Compute time: " << runtime.count() << " s\n";
    
    // Write results to output file
    std::string output_file = "algorithm_tests/pthread/results/delta-stepping_pthread_results.txt";
    std::ofstream outfile(output_file);
    outfile << "# Single-Source Shortest Path from node " << source << "\n";
    
    // Calculate statistics and write distances
    int reachable = 0;
    int max_dist = 0;
    for (int i = 0; i < graph.nodes; i++) {
        int d = dist[i].load();
        if (d == INT_MAX) {
            outfile << i << " INF\n";           // Unreachable node
        } else {
            outfile << i << " " << d << "\n";   // Reachable node with distance
            reachable++;
            max_dist = std::max(max_dist, d);
        }
    }
    outfile.close();
    
    // Display final statistics
    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Reachable nodes: " << reachable << "\n";
    std::cout << "Maximum distance: " << max_dist << "\n";
    
    // Clean up allocated resources
    delete[] dist;                              // Free distance array
    for (int i = 0; i < bucket_count; i++) {
        pthread_mutex_destroy(&bucket_mutexes[i]); // Destroy bucket mutexes
    }
    pthread_barrier_destroy(&barrier);         // Destroy synchronization barrier
    freeECLgraph(graph);                       // Free graph structure
    
    return 0;
}