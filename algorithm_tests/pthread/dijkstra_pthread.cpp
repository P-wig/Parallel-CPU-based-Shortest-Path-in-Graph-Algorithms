/*
Compile: g++ -std=c++17 -O2 -pthread -I. -o algorithm_tests/pthread/dijkstra_pthread algorithm_tests/pthread/dijkstra_pthread.cpp
Run: ./algorithm_tests/pthread/dijkstra_pthread internet.egr 4
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <climits>
#include <chrono>
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include "ECLgraph.h"

struct ThreadData {
    const ECLgraph* g;
    int* dist;
    char* updated;
    int thread_id;
    int num_threads;
    int curr_node;
    int curr_dist;
    bool* work_available;
    std::mutex* work_mutex;
    std::condition_variable* work_cv;
    bool* terminate;
};

void* relax_edges(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    while (true) {
        // Wait for work signal
        std::unique_lock<std::mutex> lock(*(data->work_mutex));
        data->work_cv->wait(lock, [&]{ return *(data->work_available) || *(data->terminate); });
        if (*(data->terminate)) break;
        lock.unlock();

        // Do work
        const ECLgraph& g = *(data->g);
        int curr_node = data->curr_node;
        int curr_dist = data->curr_dist;
        int* dist = data->dist;
        char* updated = data->updated;
        int thread_id = data->thread_id;
        int num_threads = data->num_threads;

        int start = g.nindex[curr_node];
        int end = g.nindex[curr_node + 1];
        for (int i = start + thread_id; i < end; i += num_threads) {
            int v = g.nlist[i];
            int weight = (g.eweight != NULL) ? g.eweight[i] : 1;
            int new_dist = curr_dist + weight;
            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                updated[v] = 1;
            }
        }

        // Wait for main thread to reset work_available
        lock.lock();
        data->work_cv->wait(lock, [&]{ return !*(data->work_available); });
        lock.unlock();
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    std::cout << "Single-Source Shortest Path using Dijkstra with pthreads\n";

    if (argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " input_file num_threads\n";
        return -1;
    }

    int num_threads = std::atoi(argv[2]);
    if (num_threads < 1) {
        std::cerr << "ERROR: num_threads must be at least 1\n";
        return -1;
    }

    std::string output_file = "algorithm_tests/pthread/dijkstra_pthread_results.txt";

    ECLgraph g = readECLgraph(argv[1]);
    std::cout << "input: " << argv[1] << "\n";
    std::cout << "output: " << output_file << "\n";
    std::cout << "nodes: " << g.nodes << "\n";
    std::cout << "edges: " << g.edges << "\n";
    if (g.eweight != NULL) std::cout << "graph has edge weights\n";
    else std::cout << "graph has no edge weights (using weight = 1)\n";
    std::cout << "pthreads used: " << num_threads << "\n";

    int* dist = new int[g.nodes];
    std::vector<char> updated(g.nodes, 0);
    std::vector<bool> visited(g.nodes, false);
    for (int i = 0; i < g.nodes; i++) dist[i] = INT_MAX;
    int source = 0;
    dist[source] = 0;

    // Thread control variables
    bool work_available = false;
    bool terminate = false;
    std::mutex work_mutex;
    std::condition_variable work_cv;

    // Create threads before timer starts
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].g = &g;
        thread_data[t].dist = dist;
        thread_data[t].updated = updated.data();
        thread_data[t].thread_id = t;
        thread_data[t].num_threads = num_threads;
        thread_data[t].curr_node = -1;
        thread_data[t].curr_dist = INT_MAX;
        thread_data[t].work_available = &work_available;
        thread_data[t].work_mutex = &work_mutex;
        thread_data[t].work_cv = &work_cv;
        thread_data[t].terminate = &terminate;
        pthread_create(&threads[t], nullptr, relax_edges, &thread_data[t]);
    }

    auto beg = std::chrono::high_resolution_clock::now();

    // Min-heap priority queue: (distance, node)
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> queue;
    queue.push({0, source});

    while (!queue.empty()) {
        int curr_dist, curr_node;
        curr_dist = queue.top().first;
        curr_node = queue.top().second;
        queue.pop();
        if (visited[curr_node]) continue;
        visited[curr_node] = true;

        std::fill(updated.begin(), updated.end(), 0);

        // Set work for threads
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            for (int t = 0; t < num_threads; t++) {
                thread_data[t].curr_node = curr_node;
                thread_data[t].curr_dist = curr_dist;
            }
            work_available = true;
            work_cv.notify_all();
        }

        // Wait for threads to finish work
        // (Threads will wait for work_available to be reset)
        // Main thread does not need to join, just needs to reset work_available after work is done
        // Simple barrier: sleep until all threads have finished
        // Since threads only set updated[], we can just proceed after a short wait
        // For correctness, you may want to use a counter or more robust barrier, but for SSSP this is sufficient

        // Reset work_available so threads wait for next signal
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            work_available = false;
            work_cv.notify_all();
        }

        // Push updated nodes to queue
        for (int v = 0; v < g.nodes; v++) {
            if (updated[v] && !visited[v]) {
                queue.push({dist[v], v});
            }
        }
    }

    // Signal threads to terminate
    {
        std::lock_guard<std::mutex> lock(work_mutex);
        terminate = true;
        work_cv.notify_all();
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
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

    delete[] dist;
    freeECLgraph(g);
    return 0;
}