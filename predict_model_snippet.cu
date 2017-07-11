
#include <cub/cub.cuh>

/* sum: 2x1 array */



/*
 * kernel_array_beam_slave_sincos kernel that performs the reduction manually, instead of using a library
 *
 * This kernel can be executed with any number of threads per block, as long as it's a power of 2
 * The number of threads per block is however unrelated to N
 * This kernel is also intended to be executed with only a single thread block
 */
extern "C"
__global__ void 
sincos_manual(int N, float r1, float r2, float r3, float *x, float *y, float *z, float *sum) {
    int n=threadIdx.x;
    __shared__ float tmpsum[2*block_size_x];
    tmpsum[2*n]=0.0f;
    tmpsum[2*n+1]=0.0f;

    //thread block iterates over elements 0 to N
    //values are accumulated in shared memory
    for (int i=n; i<N; i+=block_size_x) {
        float ss,cc;
        sincosf((r1*__ldg(&x[i])+r2*__ldg(&y[i])+r3*__ldg(&z[i])),&ss,&cc);
        tmpsum[2*n] += ss;
        tmpsum[2*n+1] += cc;
    }
    __syncthreads();

    //reduction loop
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (n < s) {
            tmpsum[2*n] += tmpsum[2*(n+s)];
            tmpsum[2*n+1] += tmpsum[2*(n+s)+1];
        }
        __syncthreads();
    }

    // now thread 0 will add up results
    if (n==0) {
        sum[0]=tmpsum[0];
        sum[1]=tmpsum[1];
    }

}

/*
 * kernel_array_beam_slave_sincos kernel that performs the reduction manually, instead of using a library
 *
 * This kernel can be executed with any number of threads per block, recommended to use a multiple of 32
 * The number of threads per block is unrelated to N
 * This kernel is also intended to be executed with only a single thread block
 */
extern "C"
__global__ void 
sincos_cub(int N, float r1, float r2, float r3, float *x, float *y, float *z, float *sum) {
    int n=threadIdx.x;

    // Specialize BlockReduce for a 1D block of block_size_x threads on type T
    typedef cub::BlockReduce<float, block_size_x> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float sin_sum = 0.0f;
    float cos_sum = 0.0f;

    // accumulate in thread-local registers
    for (int i=n; i<N; i+=block_size_x) {
        float ss,cc;
        sincosf((r1*__ldg(&x[i])+r2*__ldg(&y[i])+r3*__ldg(&z[i])),&ss,&cc);
        sin_sum += ss;
        cos_sum += cc;
    }
    __syncthreads();

    //reduce using CUB library
    sin_sum = BlockReduce(temp_storage).Sum(sin_sum);
    __syncthreads(); // because temp_storage is reused, sync is needed here
    cos_sum = BlockReduce(temp_storage).Sum(cos_sum);

    //thread 0 stores per-thread block result
    if (n==0) {
        sum[0]=sin_sum;
        sum[1]=cos_sum;
    }

}




// Old kernel
extern "C"
__global__ void 
kernel_array_beam_slave_sincos_original(int N, float r1, float r2, float r3, float *x, float *y, float *z, float *sum, int blockDim_2) {
  unsigned int n=threadIdx.x; //+blockDim.x*blockIdx.x;
  __shared__ float tmpsum[1000]; /* assumed to be size 2*Nx1 */
  if (n<N) {
    float ss,cc;
    sincosf((r1*__ldg(&x[n])+r2*__ldg(&y[n])+r3*__ldg(&z[n])),&ss,&cc);
    tmpsum[2*n]=ss;
    tmpsum[2*n+1]=cc;
  }
  __syncthreads();

 // Build summation tree over elements, handling case where total threads is not a power of two.
  int nTotalThreads = blockDim_2; // Total number of threads (==N), rounded up to the next power of two
  while(nTotalThreads > 1) {
    int halfPoint = (nTotalThreads >> 1); // divide by two
    if (n < halfPoint) {
     int thread2 = n + halfPoint;
     if (thread2 < blockDim.x) { // Skipping the fictitious threads >N ( blockDim.x ... blockDim_2-1 )
      tmpsum[2*n] = tmpsum[2*n]+tmpsum[2*thread2];
      tmpsum[2*n+1] = tmpsum[2*n+1]+tmpsum[2*thread2+1];
     }
    }
    __syncthreads();
    nTotalThreads = halfPoint; // Reducing the binary tree size by two
  }

  /* now thread 0 will add up results */
  if (threadIdx.x==0) {
   sum[0]=tmpsum[0];
   sum[1]=tmpsum[1];
  }
}



__device__ int
NearestPowerOf2 (int n){
  if (!n) return n;  //(0 == 2^0)

  int x = 1;
  while(x < n) {
      x <<= 1;
  }
  return x;
}

