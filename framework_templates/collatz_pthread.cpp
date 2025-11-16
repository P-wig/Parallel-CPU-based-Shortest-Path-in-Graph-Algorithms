/*
Very C style pthreading, can be more modernized with -std=c++17
compile: g++ -pthread -fpermissive -o collatz_pthread collatz_pthread.cpp

run: ./collatz_pthread 3 4000000 4
     ./collatz_pthread 3 4000000 16
*/


#include <cstdio>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <pthread.h>


// shared variables
static int global_maxlen;
pthread_mutex_t mutex;
static long threads;
static long long start;
static long long stop;

static void* collatz( void* arg ){
  const long my_rank = (long)arg;
  int local_maxlen = 0;

  // compute sequence lengths
  for ( long long i = (4 * my_rank) + start; i < stop; i += threads * 4 ) {
    long long val = i;
    int len = 1;
    do {
      len++;
      if ((val % 2) != 0) {
        val = val * 3 + 1;  // odd
      } else {
        val = val / 2;  // even
      }
    } while (val != 1);
    local_maxlen = std::max(local_maxlen, len);
  }

  pthread_mutex_lock(&mutex);
  global_maxlen = std::max(global_maxlen, local_maxlen);
  pthread_mutex_unlock(&mutex);

  return NULL;
}


int main(int argc, char* argv []){
  printf("Collatz\n");

  // check command line
  if (argc != 4) {
    std::cerr << "USAGE: " << argv[0] << " start_value stop_value threads" << std::endl;
    exit(-1);
  }
  start = atoll(argv[1]);
  stop = atoll(argv[2]);
  if (start >= stop) {
    std::cerr << "ERROR: start_value must be smaller than stop_value" << std::endl;
    exit(-1);
  }

  // check thread count
  threads = atol(argv[3]);
  if (threads < 1) {
    std::cerr << "ERROR: threads must be at least 1" << std::endl;
    exit(-1);
  }

  std::cout << "Start value: " << start << std::endl;
  std::cout << "Stop value: " << stop << std::endl;
  std::cout << "Threads: " << threads << std::endl;

  // initialize pthread variables and mutex
  pthread_t* const handle = new pthread_t[threads - 1];
  pthread_mutex_init(&mutex, NULL);

  // start time
  auto start_time = std::chrono::high_resolution_clock::now();

  // set maxlen to meaningful value
  global_maxlen = 0;

  // launch threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_create(&handle[thread], NULL, collatz, (void*)thread);
  }

  // work for master
  collatz((void*)(threads - 1));

  // join threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_join(handle[thread], NULL);
  }

  // end time
  auto end_time = std::chrono::high_resolution_clock::now();
  
  // calc
  std::chrono::duration<double> runtime = end_time - start_time;

  // print result
  std::cout << "Compute time: " << runtime.count() << " seconds" << std::endl;
  std::cout << "Max sequence length: " << global_maxlen << std::endl;

  pthread_mutex_destroy(&mutex);
  delete[] handle;
  return 0;
}
