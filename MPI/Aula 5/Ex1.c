#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOOL 1e-5
#define ITER_MAX 1000
#define NXMAX 500
#define L 1.0

double f(double x, double y){
	return x+y;
}

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
	int dims[1] = {nprocs};
	int periodic[1] = {0};
	MPI_Comm comm1d;
	int newid,nbrbottom,nbrtop;

	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periodic,1, &comm1d);
	MPI_Comm_rank(comm1d, &newid);

	MPI_Cart_shift(comm1d, 0, 1, &nbrbottom, &nbrtop);

	int firstrow,nrows;
	if(newid == 0){
		int listfirstrow[nprocs];
		int listnrows[nprocs];
		
		int nrows_p = (int)(((double)(ny-2))/((double)nprocs)+0.5);

		for(int i=0;i<nprocs-1;i++){
			listfirstrow[i] = 1+i*nrows_p;
			listnrows[i] = nrows_p;
		}
		listfirstrow[nprocs-1] = 1+(nprocs-1)*nrows_p;
		listnrows[nprocs-1] = ny-2-(nprocs-1)*nrows_p;

		MPI_Scatter(listfirstrow, 1, MPI_INT, &firstrow, 1, MPI_INT, 0, comm1d);
		MPI_Scatter(listnrows, 1, MPI_INT, &nrows, 1, MPI_INT, 0, comm1d);
	}else{
		//envia o 0 e aqui fica o 0 que e para saberem que n e ele que esta a enviar
		MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &firstrow, 1, MPI_INT, 0, comm1d);
		MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &nrows, 1, MPI_INT, 0, comm1d);
}

	double(*Vnew)[nx], (*Vold)[nx], (*myf)[nx];
	Vnew = calloc(nrows+2,sizeof(*Vnew));
	Vold = calloc(nrows+2,sizeof(*Vold));
	myf = calloc(nrows+2,sizeof(*myf));

	double h = L/((double)(nx-1));
	//inicializar matriz da funcao f
	for(int i=1;i<=nrows;i++){
		for(int j=1;j<=nx-2;j++){
			myf[i][j] = f((double)(-L/2+j*h),(double)(-L/2-(firstrow+i)*h));
		}
	}

	// inicializar condicoes fronteira
	if (newid == 0){
		for(int i=0;i<nx;i++){
			Vnew[0][i] = 0;
			Vold[0][i] = 0;
		}
	}
	if (newid == nprocs-1){
		for(int i=0;i<nx;i++){
			Vnew[nrows+1][i] = 0;
			Vold[nrows+1][i] = 0;
		}
	}
	for (int i = 0; i < nrows+2; i++)
	{
		Vnew[i][0] = 0;
		Vold[i][0] = 0;
		Vnew[i][nx-1] = 0;
		Vold[i][nx-1] = 0;
	}

	for(int iter=0;iter<ITER_MAX;iter++){
		double sums[2];
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
			printf("p%d\n", newid);
			for(int i=0;i<nrows+2;i++){
				for(int j=0;j<nx;j++){
					printf("%f ", Vnew[i][j]);
				}
				printf("\n");
			}
			
			break;
		}
		
		//sentido ascendente
		MPI_Sendrecv(Vnew[1],nx, MPI_DOUBLE, nbrtop, 0, Vnew[nrows+1], nx, MPI_DOUBLE, nbrbottom, 0, comm1d, MPI_STATUS_IGNORE);
		//MPI_Sendrecv(Vnew[nrows], nx, MPI_DOUBLE, nbrtop, 0, Vnew[0], nx, MPI_DOUBLE, nbrbottom, 0, comm1d, MPI_STATUS_IGNORE);

		//sentido descendente
		MPI_Sendrecv(Vnew[nrows], nx, MPI_DOUBLE, nbrbottom, 1, Vnew[0], nx, MPI_DOUBLE, nbrtop, 1, comm1d, MPI_STATUS_IGNORE);
		//MPI_Sendrecv(Vnew[1], nx, MPI_DOUBLE, nbrbottom, 0, Vnew[nrows+1], nx, MPI_DOUBLE, nbrtop, 0, comm1d, MPI_STATUS_IGNORE);

		//atualizar Vold
		for(int i=0;i<nrows+2;i++){
			for(int j=0;j<nx;j++){
				Vold[i][j] = Vnew[i][j];
			}
		}

	}

	free(Vnew), free(Vold), free(myf);
	MPI_Finalize();
	return 0;
}