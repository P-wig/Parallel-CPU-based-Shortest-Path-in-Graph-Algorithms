/*
Compile: mpic++ -o fractal_mpi fractal_mpi.cpp

Run: mpiexec -n 4 ./fractal_mpi 1024
*/


#include <cstdio>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "BMP24.h"
#include <mpi.h>


static void fractal(const int width, unsigned char* const pic, const int begin_row, const int end_row){
  const double scale = 0.003;
  const double xCenter = -0.663889302;
  const double yCenter =  0.353461972;

  // compute pixels of image
  const double xMin = xCenter - scale;
  const double yMin = yCenter - scale;
  const double dw = 2.0 * scale / width;

  double cy = yMin;

  for ( int row = begin_row; row < end_row; row++ ) {  // rows
    double cx = xMin;
    // remove loop-carried data dependency with function
    cy = yMin + ( dw * row );

    for (int col = 0; col < width; col++) {  // columns
      double x = cx;
      double y = cy;
      double x2, y2;
      int count = 256;
      do {
        x2 = x * x;
        y2 = y * y;
        y = 2.0 * x * y + cy;
        x = x2 - y2 + cx;
        count--;
      } while ((count > 0) && ((x2 + y2) < 5.0));
      pic[row * width + col] = (unsigned char)count;
      cx += dw;
    }
  }
}


int main(int argc, char* argv []){
  // setting up MPI
  int comm_sz, my_rank;
  MPI_Init( &argc, &argv );
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if( my_rank == 0 ){
    std::cout << "Fractal Parallel" << std::endl;
    std::cout << "Number of ranks being used is " << comm_sz << std::endl;
  }

  // check command line
  if( my_rank == 0 ) {
    if (argc != 2) {
        std::cerr << "USAGE: " << argv[0] << " image_width" << std::endl;
        MPI_Finalize();
        exit(-1);
    }
  }
  const int width = atoi(argv[1]);
  if( my_rank == 0 ) {
    if (width < 12) {
      std::cerr << "ERROR: image_width must be at least 12 pixels" << std::endl;
      MPI_Finalize();
      exit(-1);
    }
    if( width % comm_sz != 0 ){
      std::cerr << "ERROR: image_width must be divisible by number of processes" << std::endl;
      MPI_Finalize();
      exit(-1);
    }
    printf("image width: %d\n", width);
  }

  // allocate image memory
  unsigned char* pic = new unsigned char [width * width];

  // start time
  auto begin = std::chrono::high_resolution_clock::now();
  // sync for better timing
  MPI_Barrier( MPI_COMM_WORLD );

  // Block partitioning
  const int begin_row = my_rank * width / comm_sz;
  const int end_row = (my_rank + 1) * width / comm_sz;

  // execute timed code
  fractal(width, pic, begin_row, end_row);

  // make destination for gathered picture but initialize on rank 0
  unsigned char* gathered_pic_rows = nullptr;
  if( my_rank == 0 ){
    gathered_pic_rows = new unsigned char [width * width];
  }
  MPI_Barrier( MPI_COMM_WORLD );
  // gather pieces of picture from each rank
  /* MPI_Gather(
   *  void* send_buffer
   *  int send_size
   *  MPI_Datatype send_type
   *  void* receive_buffer
   *  int receive_count
   *  MPI Datatype receive_type
   *  int destination
   *  int comm
   *  */
  MPI_Gather(
          &pic[begin_row * width], //sending
          ( end_row - begin_row ) * width, //size of sending
          MPI_UNSIGNED_CHAR, //send type
          gathered_pic_rows, //receiving
          ( end_row - begin_row ) * width, //receiving size
          MPI_UNSIGNED_CHAR, //receiving type
          0, //destination
          MPI_COMM_WORLD ); //comm world

  // end time
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - begin;
  if( my_rank == 0 ) {
    std::cout << "Compute time: " << elapsed.count() << " seconds\n";
  }
  // write image to BMP file
  if ( width <= 1024 and my_rank == 0 ) {
    BMP24 bmp(0, 0, width, width);
    for (int y = 0; y < width; y++) {
      for (int x = 0; x < width; x++) {
        bmp.dot( x, y, 0x0000ff - gathered_pic_rows[y * width + x] * 0x000001 + gathered_pic_rows[y * width + x] * 0x010100 );
      }
    }
    bmp.save("fractal.bmp");
  }

  // clean up
  delete [] pic;
  if( my_rank == 0 ){
    delete[] gathered_pic_rows;
  }
  MPI_Finalize();
  return 0;
}
