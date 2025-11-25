/*
Compile: g++ -std=c++17 -O2 -pthread -I. -o algorithm_tests/pthread/delta-stepping_pthread algorithm_tests/pthread/delta-stepping_pthread.cpp
Run: ./algorithm_tests/pthread/delta-stepping_pthread internet.egr 4

Delta-Stepping Algorithm (pthreads)

- Parallel SSSP for graphs with non-negative edge weights.
- Uses sets for buckets, matching MPI logic.
- Light edge phase: repeatedly process current bucket until empty.
- Heavy edge phase: process all nodes ever in the current bucket.
- Synchronization via pthread barriers.
- Output format matches other SSSP implementations.

Arguments:
- input_file: ECLgraph format
- num_threads: number of pthreads
- delta: bucket width (positive integer)
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <climits>
#include <chrono>
#include <pthread.h>
#include <cstdlib>
#include <algorithm>
#include "ECLgraph.h"
#include <atomic>

static int threads;
static int delta;
static pthread_barrier_t barrier;
static ECLgraph g;
static std::atomic<int>* dist;
static int source = 0;
static std::vector<std::set<int>> buckets;
static int curr_bucket;
static bool done = false;
static int* finalized;
static int bucket_count;
static std::vector<std::vector<std::set<int>>> local_buckets; // [thread][bucket]
static pthread_mutex_t* node_mutexes;
static std::atomic<int> bucket_frontier(0);
static int* finalized_per_bucket; // Array to count finalized nodes per bucket
static std::atomic<bool> changed_global;
static std::set<int> S_global; // global set for bucket
static pthread_mutex_t S_mutex = PTHREAD_MUTEX_INITIALIZER; // For S_global
static std::atomic<bool> light_phase_done;
static std::atomic<bool> break_flag; // shared by all threads

static void* delta_stepping_pthread(void* arg) {
    const int tid = (int)(intptr_t)arg;
    while (true) {
        // Find next non-empty bucket
        if (tid == 0) {
            done = true;
            for (curr_bucket = 0; curr_bucket < bucket_count; curr_bucket++) {
                if (!buckets[curr_bucket].empty()) {
                    done = false;
                    break;
                }
            }
        }
        pthread_barrier_wait(&barrier);
        if (done) break;

        // Light edge phase
        static std::vector<int> curr_nodes;
        //static std::atomic<bool> break_flag;
        if (tid == 0) {
            S_global.clear();
        }
        pthread_barrier_wait(&barrier);

        while (true) {
            if (tid == 0) {
                curr_nodes.assign(buckets[curr_bucket].begin(), buckets[curr_bucket].end());
                buckets[curr_bucket].clear();
                changed_global = false;
                break_flag.store(false, std::memory_order_relaxed); // Reset at start of each iteration
            }
            pthread_barrier_wait(&barrier);

            int curr_size = curr_nodes.size();
            int block = (curr_size + threads - 1) / threads;
            int begin = tid * block;
            int end = std::min(curr_size, begin + block);

            std::set<int> local_S;
            for (int i = begin; i < end; i++) {
                int u = curr_nodes[i];
                local_S.insert(u);
                int start = g.nindex[u];
                int finish = g.nindex[u + 1];
                for (int e = start; e < finish; e++) {
                    int v = g.nlist[e];
                    int w = (g.eweight != NULL) ? g.eweight[e] : 1;
                    if (w <= delta) {
                        int new_dist = dist[u] + w;
                        pthread_mutex_lock(&node_mutexes[v]);
                        int old_dist = dist[v].load();
                        while (new_dist < old_dist) {
                            if (dist[v].compare_exchange_weak(old_dist, new_dist)) {
                                int b = new_dist / delta;
                                int frontier = bucket_frontier.load();
                                if (b < frontier) b = frontier;
                                local_buckets[tid][b].insert(v);
                                if (b == curr_bucket) changed_global = true;
                                break;
                            }
                        }
                        pthread_mutex_unlock(&node_mutexes[v]);
                    }
                }
            }
            pthread_mutex_lock(&S_mutex);
            S_global.insert(local_S.begin(), local_S.end());
            pthread_mutex_unlock(&S_mutex);

            pthread_barrier_wait(&barrier);

            if (tid == 0) {
                for (int t = 0; t < threads; t++) {
                    for (int b = 0; b < bucket_count; b++) {
                        buckets[b].insert(local_buckets[t][b].begin(), local_buckets[t][b].end());
                        local_buckets[t][b].clear();
                    }
                }
                bool bucket_empty = buckets[curr_bucket].empty();
                if (!changed_global || bucket_empty) {
                    break_flag.store(true, std::memory_order_relaxed);
                }
            }
            pthread_barrier_wait(&barrier);
            if (break_flag.load(std::memory_order_relaxed)) break;
        }

        // Heavy edge phase
        std::vector<int> S_vec;
        if (tid == 0) {
            S_vec.assign(S_global.begin(), S_global.end());
        }
        pthread_barrier_wait(&barrier);
        if (tid != 0) {
            S_vec.assign(S_global.begin(), S_global.end());
        }
        int S_size = S_vec.size();
        int block = (S_size + threads - 1) / threads;
        int begin = tid * block;
        int end = std::min(S_size, begin + block);

        for (int i = begin; i < end; i++) {
            int u = S_vec[i];
            int start = g.nindex[u];
            int finish = g.nindex[u + 1];
            for (int e = start; e < finish; e++) {
                int v = g.nlist[e];
                int w = (g.eweight != NULL) ? g.eweight[e] : 1;
                if (w > delta) {
                    int new_dist = dist[u] + w;
                    pthread_mutex_lock(&node_mutexes[v]);
                    int old_dist = dist[v].load();
                    while (new_dist < old_dist) {
                        if (dist[v].compare_exchange_weak(old_dist, new_dist)) {
                            int b = new_dist / delta;
                            int frontier = bucket_frontier.load();
                            if (b < frontier) b = frontier;
                            local_buckets[tid][b].insert(v);
                            break;
                        }
                    }
                    pthread_mutex_unlock(&node_mutexes[v]);
                }
            }
        }
        pthread_barrier_wait(&barrier);

        // Merge thread-local buckets into global buckets after heavy edge phase
        if (tid == 0) {
            for (int t = 0; t < threads; t++) {
                for (int b = 0; b < bucket_count; b++) {
                    buckets[b].insert(local_buckets[t][b].begin(), local_buckets[t][b].end());
                    local_buckets[t][b].clear();
                }
            }
        }
        pthread_barrier_wait(&barrier);

        // Finalize all nodes in S_global after all updates
        pthread_barrier_wait(&barrier);
        for (int u : S_vec) finalized[u] = 1;
        pthread_barrier_wait(&barrier);

        // Debug output: print number of unique finalized nodes in each bucket
        if (tid == 0) {
            static std::set<int> already_finalized; // persists across buckets
            std::set<int> unique_this_bucket;
            for (int u : S_vec) {
                if (already_finalized.insert(u).second) { // true if not already finalized
                    unique_this_bucket.insert(u);
                }
            }
            finalized_per_bucket[curr_bucket] += unique_this_bucket.size();
        }
        pthread_barrier_wait(&barrier);

        // Advance curr_bucket
        if (tid == 0) {
            done = true;
            for (curr_bucket = bucket_frontier; curr_bucket < bucket_count; curr_bucket++) {
                if (!buckets[curr_bucket].empty()) {
                    done = false;
                    break;
                }
            }
            bucket_frontier = curr_bucket;
        }
        pthread_barrier_wait(&barrier);
        if (done) break;
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    std::cout << "Single-Source Shortest Path using Delta-Stepping with pthreads\n";
    if (argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file num_threads\n";
        return -1;
    }
    threads = std::atoi(argv[2]);
    delta = 1000;
    if (threads < 1) {
        std::cerr << "ERROR: num_threads must be at least 1\n";
        return -1;
    }
    std::string output_file = "algorithm_tests/pthread/results/delta-stepping_pthread_results.txt";
    g = readECLgraph(argv[1]);
    std::cout << "input: " << argv[1] << "\n";
    std::cout << "output: " << output_file << "\n";
    std::cout << "nodes: " << g.nodes << "\n";
    std::cout << "edges: " << g.edges << "\n";
    if (g.eweight != NULL) std::cout << "graph has edge weights\n";
    else std::cout << "graph has no edge weights (using weight = 1)\n";
    std::cout << "pthreads used: " << threads << "\n";
    std::cout << "delta: " << delta << "\n";

    dist = new std::atomic<int>[g.nodes];
    for (int i = 0; i < g.nodes; i++) dist[i] = INT_MAX;
    dist[source] = 0;

    bucket_count = 8;
    buckets.resize(bucket_count);
    local_buckets.resize(threads);
    for (int t = 0; t < threads; t++)
        local_buckets[t].resize(bucket_count);
    buckets[0].insert(source);

    pthread_barrier_init(&barrier, nullptr, threads);
    pthread_t* handles = new pthread_t[threads - 1];
    finalized = new int[g.nodes]();
    finalized_per_bucket = new int[bucket_count]();
    node_mutexes = new pthread_mutex_t[g.nodes];
    for (int i = 0; i < g.nodes; i++) {
        pthread_mutex_init(&node_mutexes[i], nullptr);
    }

    auto beg = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < threads - 1; t++) {
        pthread_create(&handles[t], nullptr, delta_stepping_pthread, (void*)(intptr_t)t);
    }
    delta_stepping_pthread((void*)(intptr_t)(threads - 1));
    for (int t = 0; t < threads - 1; t++) {
        pthread_join(handles[t], nullptr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - beg;
    std::cout << "\ncompute time: " << runtime.count() << " s\n";

    std::ofstream outfile(output_file);
    if (!outfile) {
        std::cerr << "ERROR: could not open output file\n";
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
    std::cout << "Results written to " << output_file << "\n";
    std::cout << "Global max shortest-path: ";
    if (global_max_path == INT_MIN) std::cout << "None found\n";
    else std::cout << global_max_path << "\n";
    int reachable = 0;
    for (int i = 0; i < g.nodes; i++) {
        if (dist[i] != INT_MAX) reachable++;
    }
    std::cout << "Reachable nodes from source: " << reachable << "\n";

    // After all threads complete, in main()
    std::cout << "Per-bucket finalized node counts:\n";
    for (int b = 0; b < bucket_count; b++) {
        if (finalized_per_bucket[b] > 0)
            std::cout << "Bucket " << b << ": finalized " << finalized_per_bucket[b] << " nodes\n";
    }

    pthread_barrier_destroy(&barrier);
    delete[] handles;
    delete[] dist;
    delete[] finalized;
    delete[] finalized_per_bucket;
    delete[] node_mutexes;
    freeECLgraph(g);
    return 0;
}

/*
TODO: errors in path reduction. rewrite mpi version after this one is fixed
*/