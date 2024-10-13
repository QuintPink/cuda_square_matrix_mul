# cuda_square_matrix_mul
A small cuda c++ program for square matrix multiplication using shared memory tiling, based on the [ORLC cuda training series](https://github.com/olcf/cuda-training-series/tree/master/exercises/hw2).

---

The exercise template does not support matrix sizes that are indivisible by the block size, eventhough it makes it look like it does because of the following lines:

  > dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
and
  > if ((idx < ds) && (idy < ds)){

The seconde line can be confusing as it surrounds all working code of the kernel. However, the population of the shared memory tiles should be not be conditioned like this. Only the writing process to the output matrix should be conditioned this way. 

The code in this repository supports all matrix sizes (as long as there are enough cuda threads). 
