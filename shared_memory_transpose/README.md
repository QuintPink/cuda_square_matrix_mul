## TRANSPOSING A MATRIX WITH SHARED MEMORY

Recently I came in contact with the idea of shared memory and tiling. At first glance nothing seemed hard about using shared memory; just don't forget to call syncthreads()!. I already used memory sharing before while doing matrix multiplication. However, reading through the technical blog on [matrix transposition](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/), I realised I celebrated too early. In this document I go through what I learned about 

---

### Shared memory 
Shared memory is, as the name suggests, memory shared by all threads in a block. The memory is organised in 32 memory *banks*, where each successive 4-byte word belongs to a different bank. The maximum capacity of shared memory is configurable on a per kernel basis, in multiples of 16 KB, up to 163 KB depending on the GPU you're using. Note that the available memory is actually shared with the L1 cache. This means two things: 1. Shared memory is fast and 2. Be careful when configuring the shared memory capacity because L1 caching is automatic and thus often needed/better for unpredictable access patterns. 

Shared memory accesses are issued per warp, where every 32 consecutive threads forms a warp. (A warp can be thought of as a SIMD structure: all device instructions are executed warp-wide). If N different threads of such a warp are trying to access different data from the same bank, then they are executed serially (= big performance penalty). If threads try to access the same word (4 bytes), the fetch is broadcasted (= no performance penalty).

---

### Increasing the performance of matrix transposition

#### Rough idea of memory coalescing
In the [developer blog post on matrix transposition](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) the first priority is memory coalescing: make sure consecutive threads access consecutive global memory data such that the bus is utilized efficiently. In the naive matrix transposition program, the reading from global memory is done in a well-coalesced matter, but the writing is not. The stride between consecutive threads is too large: when doing a warp-wide write, many lines of the cache need to be invalidated/written to for only 128 bytes requested by the warp. When consecutive threads access consecutive locations, only 1 line needs to be invalidated/written back to. To remedy this, we perform tiling using shared memory. The matrix is transposed per tile (but of course with some parallelism) by reading a temporary tile of elements into shared memory. This way, the reading is split from the writing and both can be done in a coalesced manner. So why was this not possible using global memory? Because global memory is way slower: the benefit of well-coalesced memory accesses would not weigh up to the cost of extra I/O to global memory. 

#### Bank conflicts
Another big hit in performance comes from the nature of matrix transposition. Transposition requires us to access a matrix in two distinct ways since transposing a row means writing a column. Our program works in two steps: first, threads read contiguous data into rows of a shared memory tile. Second, we fill the output memory in a coalesced manner aswell, meaning a warp fills (a part of) the row of the output matrix. To fill the row, we need to read the corresponding column from the tile. In other words, the threads of a warp are reading 32 elements of the same column. 

Banks are filled such that each successive 4-byte word belongs to a different bank. Since the tile size is 32 and there are 32 banks, we have that threads reading from the same column leads to them accessing the same bank, aka we have (32-way) bank conflicts (which is the worst case in terms of bank conflicts). To solve this, we need to make sure that elements of a column are not stored in the same bank (and not somehow make elements in a row be in the same bank). As the blog explains, one can just increase the row size of the tile by 1. The 33th element of each row is never accessed but we get the benefit of having different banks for each element in a column. This is called "padding".

If you don't understand, it helps to visualise the process of "assigning" logical memory words to banks. Slides of the [the fourth episode of the cuda training series given by Robert Crovella] show this: 

without padding:
![][zero-padding-bank-assignment.png]

with padding:
![][1-padding-bank-assignment.png]

### Other findings
#### IDs of consecutive threads in a warp
To optimize memory coalescing and bank conflicts, one needs to know how threads of a grid are "assigned" to the warps. Otherwise, how does one know which threads are battling for bank access?

If the block has only one dimension, then the first warp contains thread 0 to thread 31, the second thread 32 to 63, and so on.

If the block has more dimensions, then threads are assigned in order of incrementing the first dimension (x) until the block_size.x is reached, then the second dimension (y) is incremented and we start from 0 for the first dimension. This keeps going until the second dimension reaches its block size. Then the third dimension (z) is incremented and so on. 

There is probably an easier way to explain and definitely an easy way to visualise this but I'm not bothered with that at the moment.


#### Amortization of index calculation
The blog also introduced the idea that using less threads than there are elements in a tile means the index calculation cost can be amortized and, more specifically, that it's worth it to do so.

