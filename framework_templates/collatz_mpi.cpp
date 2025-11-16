/*
Compile: mpic++ -o collatz_mpi collatz_mpi.cpp

run: mpiexec -n 4 ./collatz_mpi 3 4000000
*/


#include <cstdio>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <mpi.h>

static int collatz(const long long start, const long long stop, const int my_rank, const int comm_sz){
  int maxlen = 0;

  // compute sequence lengths
  for ( long long i = (4 * my_rank) + start; i < stop; i += comm_sz * 4 ) {
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
  // setting up MPI
  int comm_sz, my_rank;
  MPI_Init( NULL, NULL );
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if( my_rank == 0 ){
    std::cout << "Collatz Conjecture" << std::endl;
    std::cout << "Number of ranks being used is " << comm_sz << std::endl;
  }

  // check command line
  if( my_rank == 0 ) {
    if (argc != 3) {
      std::cerr << "USAGE: " << argv[0] << " start_value stop_value" << std::endl;
      MPI_Finalize();
      exit(-1);
    }
  }
  const long long start = atoll(argv[1]);
  const long long stop = atoll(argv[2]);
  if( my_rank == 0 ) {
    if (start >= stop) {
      std::cerr << "ERROR: start_value must be smaller than stop_value" << std::endl;

      MPI_Finalize();
      exit(-1);
    }
    std::cout << "Start value: " << start << std::endl;
    std::cout << "Stop value: " << stop << std::endl;
  }

  // sync for better timing
  MPI_Barrier( MPI_COMM_WORLD );
  auto start_time = std::chrono::high_resolution_clock::now();

  // execute timed code
  const int local_maxlen = collatz(start, stop, my_rank, comm_sz);
  // initialize a variable to store the global result
  int global_maxlen;
  // sync all ranks before reducing, possible redundancy
  MPI_Barrier( MPI_COMM_WORLD );
  // reduce all ranks maxlens into rank 0 maxlen
  /* int MPI_Reduce(
   *    void* input
   *    void* output
   *    int count
   *    MPI_Datatype datatype
   *    MPI_Op operation
   *    int rank
   *    MPI_Comm comm
   *    */
  MPI_Reduce( &local_maxlen, &global_maxlen, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD );

  // end time
  auto end_time = std::chrono::high_resolution_clock::now();

  //Calc
  std::chrono::duration<double> runtime = end_time - start_time;

  // print results
  if( my_rank == 0 ){
    std::cout << "Compute time: " << runtime.count() << " seconds" << std::endl;
    std::cout << "Max sequence length: " << global_maxlen << std::endl;
  }

  // closing out MPI
  MPI_Finalize();
  return 0;
}
