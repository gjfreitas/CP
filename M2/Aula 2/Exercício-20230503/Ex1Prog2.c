#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
int main(int argc, char *argv[])
{
  int n, myid, numprocs;
  int *bloco;
  double MPI_Wtime(), MPI_Wtick();
  double starttime, endtime;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  /* Queremos repetir com vários tamanhos de mensagem. */ 
  while (1) {
    /* O master lê o nº de inteiros na mensagem, n. */
    usleep(3E5);
    if (myid == 0) {
      printf("Número de inteiros a enviar: (0 p/ sair) ");
      (void)! scanf("%d",&n);
    }
  
    /* Valor de n é distribuido por todos. */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) break; /* Todos terminam.*/

    /* Todos definem um vetor de inteiros.
     * Não havia necessidade nenhuma de os processos com myid > 1
     * fazerem isto. Só estamos a usar memória sem necessidade.*/
    bloco = (int*)malloc(n * sizeof(int));
    /* Todos começam a contar o tempo no mesmo instante. */
    MPI_Barrier(MPI_COMM_WORLD);
    starttime = MPI_Wtime();

    if (myid == 0) {
      for (int i=0; i < n; ++i) bloco[i]=2*i;
      /* É preciso criar o buffer. */
      int buffer_size = (MPI_BSEND_OVERHEAD + n*sizeof(int));
      char* buffer = malloc(buffer_size);
      MPI_Buffer_attach(buffer, buffer_size);
      MPI_Bsend(bloco, n, MPI_INT, 1, 0, MPI_COMM_WORLD);
      endtime=MPI_Wtime();
      printf("Sou o %d. Passaram aproximadamente %f segundos "
             "desde que comecei a enviar o bloco "
             "e agora estou pronto para continuar.\n", myid, endtime-starttime);
      MPI_Buffer_detach(&buffer, &buffer_size);
      free(buffer);
    }
    else if (myid == 1) {
      usleep(5E6);
      MPI_Recv(bloco, n, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      printf("Sou o %d. Estive 5 segundos a dormir e acabei agora de "
             "receber o bloco.\n", myid);
    }
    else {
      /* Não se faz nada. */
    }

     free(bloco);
     /* Não quero que o 0 comece a pedir o n antes de o 1 fazer o output. */
     MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}
