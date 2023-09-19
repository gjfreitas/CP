// adapted from "A “Hands-on” Introduction to OpenMP" Tim Mattson Intel Corp. timothy.g.mattson@intel.com 

#include <stdio.h>
#include <omp.h>

int main ()  
{   
    //#pragma omp parallel num_threads(4) // 2.2
    int num_threads = 40;
    omp_set_num_threads(num_threads); // 2.3
    #pragma omp parallel
    {
    int ID = omp_get_thread_num();

    printf("Hello (%d) ", ID);
    printf("World (%d) \n", ID);
    }
}
