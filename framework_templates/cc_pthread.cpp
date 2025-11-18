/*
compile: g++ -pthread -o cc_pthread cc_pthread.cpp

Run: ./cc_pthread amazon0601.egr 28
     ./cc_pthread internet.egr 28
     ./cc_pthread citationCiteseer.egr 28


some graphs cannot verify, exit with no error code, possible memory constraint.
*/


#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <set>
#include <chrono>
#include "ECLgraph.h"
#include <pthread.h>


static long threads;
pthread_barrier_t barrier;
pthread_mutex_t repeat_mutex; // Mutex for repeat flag
static bool go_again;
static bool repeat;
static ECLgraph g;
static int* old_label;
static int* new_label;

static void* cc( void* arg ) {
  const long long my_rank = (long long)arg;
  const int node_block = g.nodes / threads;
  const int start = my_rank * node_block;
  const int stop = (my_rank + 1) * node_block;
  
  // initialize labels
  for (int v = start; v < stop; v++) {
      old_label[v] = v;
  }

  // Place a barrier in the cc function right before the do-while loop
  pthread_barrier_wait(&barrier);
  // repeat until all nodes' labels have converged
  do {
      bool local_repeat = false; //this was global, not sure why

      // go over all nodes
      for (int v = start; v < stop; v++) {
        int my_label = old_label[v];

        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
            int n = g.nlist[i]; // Neighbor
            if (my_label > old_label[n]) { // Corrected condition (propagate smaller label)
                my_label = old_label[n];
                local_repeat = true; // A change occurred
            }
        }
        new_label[v] = my_label;
    }

      pthread_barrier_wait(&barrier);

      // Update the global repeat flag safely
      pthread_mutex_lock(&repeat_mutex);
      repeat = repeat || local_repeat;
      pthread_mutex_unlock(&repeat_mutex);

      // swap the two label arrays
      if (my_rank == 0) {
        std::swap(old_label, new_label);

        go_again = repeat;
        repeat = false;
      }

      // barrier right after swapping the old- and new-label pointers
      pthread_barrier_wait(&barrier);

  } while (go_again);

  return NULL;
}


static void verify(const int v, const int id, const int* const nidx, const int* const nlist, int* const label){
  if (label[v] >= 0) {
    if (label[v] != id) {
      std::cerr << "ERROR: found incorrect ID value\n\n";  
      exit(-1);
    }
    label[v] = -1;
    for (int i = nidx[v]; i < nidx[v + 1]; i++) {
      verify(nlist[i], id, nidx, nlist, label);
    }
  }
}


int main(int argc, char* argv []) {
  std::cout << "Connected components via pthreads\n";

  // check command line
  if (argc != 3) {
    std::cout << "USAGE: " <<  argv[0] << " input_file\n"; 
    exit(-1);
  }

  threads = atol(argv[2]);
  if (threads < 1) {
    std::cout <<  "ERROR: threads must be at least 1\n"; 
    exit(-1);
  }
  std::cout << "threads: " << threads << "\n";

  // read input
  g = readECLgraph(argv[1]);
  std::cout << "input: " << argv[1] << "\n";
  std::cout << "nodes: " << g.nodes << "\n";
  std::cout << "edges: " << g.edges << "\n";

  // allocate arrays
  old_label = new int [g.nodes];
  new_label = new int [g.nodes];

  // initialize pthreads and barrier
  pthread_t* const handle = new pthread_t[threads - 1];
  pthread_barrier_init(&barrier, NULL, threads);

  // start time
  auto beg = std::chrono::high_resolution_clock::now();

  // launch threads
  for (long thread = 0; thread < threads - 1; thread++) {
      pthread_create(&handle[thread], NULL, cc, (void*)thread);
  }

  // work for master
  cc((void*)(threads - 1));

  // join threads
  for (long thread = 0; thread < threads - 1; thread++) {
      pthread_join(handle[thread], NULL);
  }

  // end time
  auto end = std::chrono::high_resolution_clock::now();

  // calc
  std::chrono::duration<double> runtime = end - beg;
  std::cout << "compute time: " << runtime.count() << " s\n";

  // determine number of connected components
  std::set<int> s;
  for (int v = 0; v < g.nodes; v++) {
    s.insert(new_label[v]);
  }
  std::cout << "number of connected components: " << s.size() << "\n";

  // verify result
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      const int n = g.nlist[i];
      if (new_label[n] != new_label[v]) {
        std::cerr << "ERROR: found adjacent nodes belonging to different components\n\n";  
        exit(-1);
      }
    }
  }
  for (int v = 0; v < g.nodes; v++) {
    const int lv = new_label[v];
    if ((lv < 0) || (lv >= g.nodes)) {
      std::cerr << "ERROR: found out-of-bounds component number\n\n";  
      exit(-1);
    }
    const int lbl = new_label[lv];
    if (lbl != lv) {
      std::cerr << "ERROR: representative is not in component\n\n";  
      exit(-1);
    }
  }
  int count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (new_label[v] >= 0) {
      count++;
      verify(v, new_label[v], g.nindex, g.nlist, new_label);
    }
  }
  if (s.size() != count) {
    std::cerr <<  "ERROR: component IDs are not unique\n\n";  
    exit(-1);
  }
  std::cout << "verification passed\n";

  // clean up
  pthread_barrier_destroy(&barrier);
  freeECLgraph(g);
  delete [] old_label;
  delete [] new_label;
  return 0;
}
