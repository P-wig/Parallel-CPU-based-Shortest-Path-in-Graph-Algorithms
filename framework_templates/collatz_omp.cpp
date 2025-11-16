/*
Compile: g++ -fopenmp -o collatz_omp collatz_omp.cpp

Run: ./collatz_omp 3 4000000 4
     ./collatz_omp 3 4000000 8

8 threads takes longer than 4 threads
*/


#include <cstdio>
#include <algorithm>
#include <chrono>
#include <iostream>


static int collatz( const long long start, const long long stop, const int threads ){
  int maxlen = 0;

  // compute sequence lengths
# pragma omp parallel for default( none ) num_threads( threads ) \
            reduction( max:maxlen ) shared( start, stop ) schedule( runtime )
  for (long long i = start; i < stop; i += 4) {
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
    maxlen = std::max(maxlen, len);
  }

  return maxlen;
}


int main(int argc, char* argv []){
  std::cout << "Collatz" << std::endl;

  // check command line
  if (argc != 4) {
    std::cerr << "USAGE: " << argv[0] << " start_value stop_value thread_count" << std::endl;
    exit(-1);
  }
  const long long start = atoll(argv[1]);
  const long long stop = atoll(argv[2]);
  if (start >= stop) {
    std::cerr << "ERROR: start_value must be smaller than stop_value" << std::endl;
    exit(-1);
  }

  std::cout << "start value: " << start << std::endl;
  std::cout << "stop value: " << stop << std::endl;

  const int threads = atoi(argv[3]);
  if (threads < 1) {
    std::cerr << "ERROR: thread_count must be at least 1" << std::endl;
    exit(-1);
  }

  std::cout << "thread count: " << threads << std::endl;

  // start time
  auto start_time = std::chrono::high_resolution_clock::now();

  // execute timed code
  const int maxlen = collatz(start, stop, threads);

  // end time
  auto end_time = std::chrono::high_resolution_clock::now();

  // calc
  std::chrono::duration<double> runtime = end_time - start_time;
  std::cout << "compute time: " << runtime.count() << " second" << std::endl;

  // print result
  std::cout << "max sequence length: " << maxlen << std::endl;
  return 0;
}
