// COMO CORRER O PROGRAMA
// ----------------------
// COMPILAR: mpicc calculo_Pi.c -o calculo_Pi
// CORRER:   mpiexec -n 4 calculo_Pi
// ----------------------

// Secção para dar includes
#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Função main
int main (int argc, char *argv[]){
    
    // Definir a variavel para o rank do processo
    int rank_processo;
    // Definir a variável que contem o número total de processos
    int n_processos;
    // Definir o número de bins para o calculo de pi
    int n_bins;
    // Pi teórico com 25 difitos
    double pi_teorico = 3.141592653589793238462643;
    // Definir uma variável para armazenar a soma de todas as areas de todos os bins, ou seja, o nosso pi obtido
    double pi_obtido;
    // Definir a largura do bin
    double largura_bin;
    // Definir a area de um bin
    double area_total_bins;
    // Definir o x que é o valor da formula para calcular o pi -> 4 / (1 + x²)
    double x;
    // Variável para armazenar os pi's de todos os processos
    double pi;

    // Inicialização do MPI environment
    MPI_Init(&argc, &argv);
    // Obter o número de processos totais
    MPI_Comm_size(MPI_COMM_WORLD, &n_processos);
    // Obter o rank do processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_processo);

    // Ciclo while que permite a introdução de vários valores para o calculo do Pi
    while(1){
        //printf("CHega aqui ???\n");
        // So o primeiro processo é que vai guardar a informação
        if(rank_processo == 0){

            // Pedir ao utilizador o numero de bins que quer
            printf("Introduza o número de bins(intervalos) desejados: \n");
            // Guardamos o valor introduzido pelo utilizador -> o espaço e para evitar que o utilziador introduza um caracter antes do valor
            scanf(" %d", &n_bins);

        }
        // Vamos espalhar a informação por todos os processos, neste caso: 
        // n_bins, só um elemento, o seu tipo, o processo que esta a enviar, o contexto
        MPI_Bcast(&n_bins, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        // Condição de paragem do ciclo while
        if(n_bins == 0){
            // Terminar o ciclo while
            break;
        }

        // Definir a largura do bin -> fazer estas convercoes por causa das definições
        largura_bin = 1.0 / (double)n_bins;
        // Inicializar a soma total dos bins
        area_total_bins = 0.0;
        // Definir o ciclo for que percorre todos os bins -> os processos calculacarrm cada bin sequencialmente
        for(int i = rank_processo; i < n_bins; i += n_processos){

            // Calcular o x da formula, que é o meio do retangulo
            x = largura_bin * ((double)i + 0.5);
            // Calcular a area
            area_total_bins += 4.0 / (1.0 + x*x);

        }
        // Pi obtido
        pi_obtido = area_total_bins * largura_bin;
        // Guardar os pi's todos na variável pi no processo 0
        MPI_Reduce(&pi_obtido, &pi, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD);
        // Print para ilustra a contribuição de cada processo
        printf("\nO processo %d/%d calculou o pi_obtido = %f", rank_processo, n_processos, pi_obtido);
        MPI_Barrier(MPI_COMM_WORLD);
        // Printar o resutlado para o processo 0 apenas, que é onde o resutladoe esta guardado
        if(rank_processo == 0){
            printf("O valor de Pi aproxiado é: %f \nO erro é de %f%%.\n", pi, fabs(pi_teorico - pi) / pi_teorico);
        }

    }

    // Termianr o MPI environment.
    MPI_Finalize();
    // Finalizar a função main
    return 0;
}