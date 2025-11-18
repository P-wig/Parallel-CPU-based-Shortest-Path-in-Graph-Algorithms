/*
Compile: g++ -fopenmp -o cc_omp cc_omp.cpp

Run: ./cc_omp amazon0601.egr
     ./cc_omp internet.egr
     ./cc_omp citationCiteseer.egr


some graphs cannot verify, exit with no error code, possible memory constraint.
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <set>
#include <chrono>
#include "ECLgraph.h"


static void cc(const ECLgraph g, int* old_label, int* new_label){
  // initialize labels
  bool go_again;
  bool repeat = false;

# pragma omp parallel default( none ) shared( g, old_label, new_label, repeat, go_again )
  {
#   pragma omp for
    for (int v = 0; v < g.nodes; v++) {
        old_label[v] = v;
    }
    // repeat until all nodes' labels have converged
    do {
        // go over all nodes
#       pragma omp for schedule( dynamic, 64 ) reduction( ||:repeat )
        for (int v = 0; v < g.nodes; v++) {
            const int beg = g.nindex[v];  // beginning of adjacency list
            const int end = g.nindex[v + 1];  // end of adjacency list
            int my_label = old_label[v];
            for (int i = beg; i < end; i++) {
                const int n = g.nlist[i];  // neighbor
                const int nbor_label = old_label[n];
                // update my label if smaller
                if (my_label < nbor_label) {
                    my_label = nbor_label;
                    repeat = true;
                }
            }
            new_label[v] = my_label;
        }

        // swap the two label arrays
#       pragma omp single
        {
            go_again = repeat;
            repeat = false;
            std::swap(old_label, new_label);
        }
    } while (go_again);
  }
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


int main(int argc, char* argv []){
  std::cout << "Connected components via OpenMP\n";

  // check command line
  if (argc != 2) {
    std::cerr << "USAGE: " << argv[0] << " input_file\n"; 
    exit(-1);
  }

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  std::cout << "input: " << argv[1] << "\n";
  std::cout << "nodes: " << g.nodes << "\n";
  std::cout << "edges: " << g.edges << "\n";

  // allocate arrays
  int* const old_label = new int [g.nodes];
  int* const new_label = new int [g.nodes];

  // start time
  auto beg = std::chrono::high_resolution_clock::now();

  // execute timed code
  cc(g, old_label, new_label);

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
  freeECLgraph(g);
  delete [] old_label;
  delete [] new_label;
  return 0;
}
