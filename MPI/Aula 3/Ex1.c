#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define min(a,b) (((a)<(b))?(a):(b))

int main(int argc, char *argv[]){
    int manager_rank = 0;
    int nrows = 10;
    int ncols = 10;

    int b[ncols];
    int nprocs, myrank;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MPI_Status status;

    int tag_term = nrows;        
    int C[nrows];

    if (myrank == manager_rank){


        int A[nrows][ncols];
        for(int i=0; i<nrows; i++){
            for(int j=0; j<ncols; j++){
                A[i][j] = j+i*ncols;
            }
        }

        for(int i=0; i<ncols; i++){
            b[i] = i;
        }
        MPI_Bcast(b, ncols, MPI_INT, manager_rank, MPI_COMM_WORLD); // do vetor b

        int next_row = 0;
        for (int i=1; i <= min(nprocs-1,nrows); i++){
            MPI_Send(A[next_row], ncols, MPI_INT, i, next_row, MPI_COMM_WORLD);
            next_row++;
        }

        int ans;

        for(int i=0; i<nrows; i++){
            MPI_Recv(&ans, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // recebe uma resposta
            int sender = status.MPI_SOURCE;
            int row_index = status.MPI_TAG;
            C[row_index] = ans;
            
            if(next_row < nrows){
                MPI_Send(A[next_row], ncols, MPI_INT, sender, next_row, MPI_COMM_WORLD); // envia a proxima linha
                next_row++;
            }
            else{
                MPI_Send(MPI_BOTTOM, 0, MPI_INT, sender, tag_term, MPI_COMM_WORLD); // envia um sinal de termino
            }
        }

        for(int i =0; i<nrows; i++){
            printf("C[%d] = %d \n", i,C[i]);
        }

        }
    else{
        int row[ncols];
        MPI_Bcast(b, ncols, MPI_INT, manager_rank, MPI_COMM_WORLD); // recebe o vetor b

        if (myrank <= nrows){
            while(1){
                MPI_Recv(row, ncols, MPI_INT, manager_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // recebe uma linha
                int row_index = status.MPI_TAG;
                if(row_index == tag_term){
                    break;
                }
                else{
                    int sum = 0;
                    for(int i=0; i<ncols; i++){
                        sum += row[i]*b[i];
                    }
                    MPI_Send(&sum, 1, MPI_INT, manager_rank, row_index, MPI_COMM_WORLD); // envia a resposta
                }
            }
        }

    }

    printf("Process %d finished\n", myrank);

    MPI_Finalize();

    return 0;

}