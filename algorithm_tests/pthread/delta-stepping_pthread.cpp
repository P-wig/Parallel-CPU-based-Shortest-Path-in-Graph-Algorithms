/*
Compile: g++ -std=c++17 -O2 -pthread -I. -o algorithm_tests/pthread/delta-stepping_pthread algorithm_tests/pthread/delta-stepping_pthread.cpp
Run: ./algorithm_tests/pthread/delta-stepping_pthread internet.egr 4

Delta-Stepping Algorithm (pthreads) - Clean Implementation
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

struct ThreadData {
    int thread_id;
    int num_threads;
    ECLgraph* graph;
    std::atomic<int>* dist;
    std::vector<std::queue<int>>* buckets;
    std::vector<pthread_mutex_t>* bucket_mutexes;
    pthread_barrier_t* barrier;
    std::atomic<int>* current_bucket;
    std::atomic<bool>* done;
    std::atomic<int>* active_threads;
    int delta;
    int bucket_count;
};

void* delta_stepping_worker(void* arg) {
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
    
    while (!done->load()) {
        // Process current bucket completely
        while (true) {
            int curr_b = current_bucket->load();
            if (curr_b >= bucket_count) {
                break;
            }
            
            // Try to get work from current bucket
            std::vector<int> local_work;
            pthread_mutex_lock(&bucket_mutexes[curr_b]);
            
            // Take up to 3 nodes to reduce contention but ensure progress
            for (int i = 0; i < 3 && !buckets[curr_b].empty(); i++) {
                local_work.push_back(buckets[curr_b].front());
                buckets[curr_b].pop();
            }
            
            pthread_mutex_unlock(&bucket_mutexes[curr_b]);
            
            if (local_work.empty()) {
                break; // No more work in current bucket
            }
            
            active_threads->fetch_add(1);
            
            // Process work
            for (int u : local_work) {
                int u_dist = dist[u].load();
                
                for (int e = g->nindex[u]; e < g->nindex[u + 1]; e++) {
                    int v = g->nlist[e];
                    int weight = (g->eweight != nullptr) ? g->eweight[e] : 1;
                    int new_dist = u_dist + weight;
                    
                    int old_dist = dist[v].load();
                    while (new_dist < old_dist) {
                        if (dist[v].compare_exchange_weak(old_dist, new_dist)) {
                            int bucket_id = std::min(new_dist / delta, bucket_count - 1);
                            
                            pthread_mutex_lock(&bucket_mutexes[bucket_id]);
                            buckets[bucket_id].push(v);
                            pthread_mutex_unlock(&bucket_mutexes[bucket_id]);
                            break;
                        }
                    }
                }
            }
            
            active_threads->fetch_sub(1);
        }
        
        // Single barrier - wait for all threads to finish current bucket
        pthread_barrier_wait(barrier);
        
        // Only thread 0 handles bucket advancement
        if (tid == 0) {
            int curr_b = current_bucket->load();
            
            // Check if current bucket has new work
            pthread_mutex_lock(&bucket_mutexes[curr_b]);
            bool current_has_work = !buckets[curr_b].empty();
            pthread_mutex_unlock(&bucket_mutexes[curr_b]);
            
            if (current_has_work) {
                // Stay on current bucket - it has more work
            } else {
                // Find next bucket with work
                int next_bucket = curr_b + 1;
                while (next_bucket < bucket_count) {
                    pthread_mutex_lock(&bucket_mutexes[next_bucket]);
                    bool has_work = !buckets[next_bucket].empty();
                    pthread_mutex_unlock(&bucket_mutexes[next_bucket]);
                    
                    if (has_work) break;
                    next_bucket++;
                }
                
                if (next_bucket >= bucket_count) {
                    // Final check - make sure no threads are still working
                    if (active_threads->load() == 0) {
                        done->store(true);
                    }
                } else {
                    current_bucket->store(next_bucket);
                }
            }
        }
        
        // Single barrier to sync bucket advancement decision
        pthread_barrier_wait(barrier);
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
    int delta = 50;
    int bucket_count = (7000 / delta) + 10;
    int source = 0;
    
    std::cout << "Delta-Stepping with pthreads (Clean)\n";
    std::cout << "Nodes: " << graph.nodes << "\n";
    std::cout << "Edges: " << graph.edges << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Delta: " << delta << "\n";
    
    // Initialize distances
    std::atomic<int>* dist = new std::atomic<int>[graph.nodes];
    for (int i = 0; i < graph.nodes; i++) {
        dist[i].store(INT_MAX);
    }
    dist[source].store(0);
    
    // Initialize buckets
    std::vector<std::queue<int>> buckets(bucket_count);
    std::vector<pthread_mutex_t> bucket_mutexes(bucket_count);
    
    for (int i = 0; i < bucket_count; i++) {
        pthread_mutex_init(&bucket_mutexes[i], nullptr);
    }
    
    buckets[0].push(source);
    
    // Initialize synchronization
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, num_threads);
    std::atomic<int> current_bucket(0);
    std::atomic<bool> done(false);
    std::atomic<int> active_threads(0);
    
    // Create thread data
    std::vector<ThreadData> thread_data(num_threads);
    for (int i = 0; i < num_threads; i++) {
        thread_data[i] = {i, num_threads, &graph, dist, &buckets, &bucket_mutexes,
                         &barrier, &current_bucket, &done, &active_threads, delta, bucket_count};
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create threads
    std::vector<pthread_t> threads(num_threads - 1);
    for (int i = 0; i < num_threads - 1; i++) {
        pthread_create(&threads[i], nullptr, delta_stepping_worker, &thread_data[i]);
    }
    
    // Main thread also works
    delta_stepping_worker(&thread_data[num_threads - 1]);
    
    // Wait for threads
    for (int i = 0; i < num_threads - 1; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;
    
    std::cout << "Compute time: " << runtime.count() << " s\n";
    
    // Write results
    std::string output_file = "algorithm_tests/pthread/results/delta-stepping_pthread_results.txt";
    std::ofstream outfile(output_file);
    outfile << "# Single-Source Shortest Path from node " << source << "\n";
    
    int reachable = 0;
    int max_dist = 0;
    for (int i = 0; i < graph.nodes; i++) {
        int d = dist[i].load();
        if (d == INT_MAX) {
            outfile << i << " INF\n";
        } else {
            outfile << i << " " << d << "\n";
            reachable++;
            max_dist = std::max(max_dist, d);
        }
    }
    outfile.close();
    
    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Reachable nodes: " << reachable << "\n";
    std::cout << "Maximum distance: " << max_dist << "\n";
    
    // Cleanup
    delete[] dist;
    for (int i = 0; i < bucket_count; i++) {
        pthread_mutex_destroy(&bucket_mutexes[i]);
    }
    pthread_barrier_destroy(&barrier);
    freeECLgraph(graph);
    
    return 0;
}