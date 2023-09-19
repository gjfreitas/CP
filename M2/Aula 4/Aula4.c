#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOOL 1e-5
#define ITER_MAX 1000
#define NXMAX 500
#define L 1.0


int main(int argc, char *argv[]){

	int nprocs;
	int myid;
	int nx, ny;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if(myid == 0){
		printf("Numero de pontos (max %d,0 para sair): ",NXMAX);
		scanf(" %d", &nx);
	}

	MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	ny=nx;

	if(nx == 0) {
		MPI_Finalize();
		return 0;
	}
	else if (nx > NXMAX){
		printf("Numero de pontos superior ao permitido\n");
		MPI_Finalize();
		return 1;
	}

	int ndims = 1;
	int dims = [nprocs];
	int periodic = [0];
	MPI_Comm comm1d;
	int newid,nbrbottom,nbrtop;

	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periodic,1, &comm1d);
	MPI_Comm_rank(comm1d, &newid);

	MPI_Cart_shift(comm1d, 0, 1, &nbrbottom, &nbrtop);

	int firstrow,nrows;
	if newid == 0{
		int listfirstrow[nprocs];
		int listnrows[nprocs];
		for(int i=0; i<nprocs; i++){
			firstrow[i] = 0;
		}
		MPI_Scatter(listfirstrow, 1, MPI_INT, &firstrow, 1, MPI_INT, 0, comm1d);
	}else{
		MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &firstrow, 1, MPI_INT, 0, comm1d);
	}

	int(*Vnew)[nx];
	Vnew = calloc(nrows+2,sizeof(*Vnew));

	//inicializar matriz da funcao f
	// inicializar condicoes fronteira

	double h=L/((double)(nx-1));

	for(int iter=0;iter<ITER_MAX;iter++){
		for(int i=1;i<=nrows;i++){
			for(int j=1;j<=nx-2;j++){
				Vnew[i][j] = 0.25*(Vold[i-1][j]+Vold[i+1][j]+Vold[i][j-1]+Vold[i][j+1]-h*h*myf[i][j]);
				sums[0] += (Vnew[i][j]-Vold[i][j])*(Vnew[i][j]-Vold[i][j]);
				sums[1] += Vnew[i][j]*Vnew[i][j];
			}
		}
		double global_sums[2];
		// troca de informacao entre processos
		MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, comm1d);
		if(global_sums[0]/global_sums[1] < TOOL){
			//escrever

			break;
		}
		
		//sentido ascendente
		MPI_Sendrecv(Vnew[nrows], nx, MPI_DOUBLE, nbrtop, 0, Vnew[0], nx, MPI_DOUBLE, nbrbottom, 0, comm1d, MPI_STATUS_IGNORE);

	}

	MPI_Finalize();
	return 0;
}
