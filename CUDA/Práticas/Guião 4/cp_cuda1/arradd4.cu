#include  <stdio.h> 
#include  <time.h> 

#define  NROWS (4) 
#define  NCOLS (4) 
#define  SIZE (NROWS*NCOLS) 


int compare(float *a1, float *a2);


//  Kernel definition, see also section 2.1 of NVIDIA CUDA Programming Guide 
__global__  void arrAdd(float *A, float *B, float *C) 
{ 
    // TODO: determine id
    int id;

    id =( blockIdx.x * blockDim.x + threadIdx.x)*NCOLS +( blockIdx.y * blockDim.y + threadIdx.y);

    if(id < SIZE)
    {
        C[id] = A[id] + B[id]; 
    }
} 

int  main(void) 
{ 
    float A[SIZE], B[SIZE], D[SIZE], H[SIZE];
    float *devPtrA; 
    float *devPtrB; 
    float *devPtrD; 
    int memsize = SIZE * sizeof(float); 
    float devExecTime;

    cudaSetDevice(1);    // Select GPU device (can be 0 to 1)

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize arrays
    srand (time(NULL));
    for(int i=0; i < SIZE; i++) 
    {
        A[i]=rand() % 100;
        B[i]=rand() % 100;
    }

    printf("Starting HOST...\n");

    for(int i=0; i < NCOLS; i++)
    {
        for(int j=0; j < NROWS; j++)
        {
            int id = j + i * NROWS;
            H[id] = A[id] + B[id];
        }
    }

    // Allocate device memory for A, B and D arrays
    cudaMalloc((void**)&devPtrA, memsize); 
    cudaMalloc((void**)&devPtrB, memsize); 
    cudaMalloc((void**)&devPtrD, memsize); 

    printf("Starting DEVICE...\n");
    cudaEventRecord(start);

    // Copy data (data to process) from host to device (from CPU to GPU)
    cudaMemcpy(devPtrA, A, memsize,  cudaMemcpyHostToDevice); 
    cudaMemcpy(devPtrB, B, memsize,  cudaMemcpyHostToDevice); 

    // __global__ functions are called:  Func <<< dim grid, dim block >>> (parameter); 
    dim3 dimBlock(2,2);
    dim3 dimGrid((NROWS + dimBlock.x -1) / dimBlock.x, (NCOLS + dimBlock.y - 1)/ dimBlock.y);

    // Execute the Kernel 
    arrAdd <<<dimGrid, dimBlock>>> (devPtrA,  devPtrB, devPtrD); 

    // Copy data from device (results) back to host 
    cudaMemcpy(D, devPtrD, memsize,  cudaMemcpyDeviceToHost); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&devExecTime, start, stop); //Exec time = elapsed time

    // Show results
    printf("     A      B       D      H\n");
    for (int i=0; i < SIZE; i++) 
    {
        printf("%2d: %4.1f + %4.1f = %5.1f [%5.1f]\n", i, A[i], B[i], D[i], H[i]); 
    }

    printf("\nOutput arrays (H/D) are %s\n", compare(D, H) == 1 ? "EQUAL" : "DIFFERENT");

    printf("\nDevice execution time [ms]: %7.4f\n", devExecTime);

    // Free device memory
    cudaFree(devPtrA); 
    cudaFree(devPtrB); 
    cudaFree(devPtrD); 
} 

int compare(float *a1, float *a2)
{
    int i, j, equal = 1;
    for(j=0; (j < NROWS) && equal; j++)
    {
        for(i=0; (i < NCOLS) && equal; i++)
        {
            int id = i + j * NCOLS;
            if(a1[id] != a2[id])
                equal = 0;
        }
    }
    return equal;
}


