/*
adapted from "A “Hands-on” Introduction to OpenMP" Tim Mattson Intel Corp. timothy.g.mattson@intel.com 

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

The is the original sequential program.  It uses the timer
from the OpenMP runtime library

History: Written by Tim Mattson, 11/99.

*/

#include <stdio.h>
#include <omp.h>

static long num_steps = 1000000000;
double step;

int main ()
{
	int i;
	double x, pi, sum = 0.0;
	double start_time, run_time;
	

	step = 1.0/(double) num_steps;

			
	start_time = omp_get_wtime();

	for (i=0; i< num_steps; i++){
		x = (i+0.5)*step;
		sum = sum + 4.0/(1.0+x*x);
	}

	pi = step * sum;
	run_time = omp_get_wtime() - start_time;
	printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,run_time);

	/////////////////////////////////////////////////////////////////////////////////
	double start_time_omp, run_time_omp;

	step = 1.0/(double) num_steps;
	pi = 0;

	printf("\nWith #pragma omp parallel (without using omp for)");

	int nth = 10;
	float result[nth];

	start_time_omp = omp_get_wtime();

	#pragma omp parallel num_threads(nth) private(i,x)
	{
		int id = omp_get_thread_num();
		double sum_omp = 0.0;

		for (i=id; i< num_steps; i+=nth){
			x = (i+0.5)*step;
			sum_omp = sum_omp + 4.0/(1.0+x*x);
		}
		result[id] = sum_omp;
	}

    float soma = 0.0;
	for(i=0; i<nth; i++){
		soma += result[i];
	}
	pi = step * soma;
	run_time_omp = omp_get_wtime() - start_time_omp;
	printf("\npi with %ld steps is %lf in %lf seconds\n",num_steps,pi,run_time_omp);
	printf("Speed up with %i threads is %lfx\n", nth, run_time/run_time_omp);

	/////////////////////////////////////////////////////////////////////////////////
	printf("\nWith omp for");
	double start_time_omp_for, run_time_omp_for;
	sum = 0.0;

    step = 1.0/(double) num_steps;
	nth = 10;

	start_time_omp_for = omp_get_wtime();

	float result_for[nth];

    #pragma omp parallel num_threads(nth) private(i,x)
	{
		int id = omp_get_thread_num();
		double sum_omp = 0.0;
	#pragma omp for
		for (i=0; i< num_steps; i++){
			x = (i+0.5)*step;
			sum_omp = sum_omp + 4.0/(1.0+x*x);
		}
		result_for[id] = sum_omp;
	}

	soma = 0.0;
	for(i=0; i<nth; i++){
		soma += result_for[i];
	}
	pi = step * soma;
    run_time_omp_for = omp_get_wtime() - start_time_omp_for;
    printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,run_time_omp_for);
	printf("Speed up with %i threads is %lfx\n", nth, run_time/run_time_omp_for);



	/////////////////////////////////////////////////////////////////////////////////
	printf("\nWith omp for and reduction");

	double start_time_for, run_time_for;

    sum = 0.0;

    step = 1.0/(double) num_steps;
	nth = 10;

    start_time_for = omp_get_wtime();

    #pragma omp parallel for num_threads(nth) private(x) reduction(+:sum)
    for (i=0; i< num_steps; i++){
        x = (i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }

    pi = step * sum;
    run_time_for = omp_get_wtime() - start_time_for;
    printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,run_time_for);
	printf("Speed up with %i threads is %lfx\n", nth, run_time/run_time_for);
}
 

