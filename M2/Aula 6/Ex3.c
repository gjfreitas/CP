// COMO CORRER O PROGRAMA - 2D
// ----------------------
// COMPILAR: mpicc Parte_3.c -o Parte_3
// CORRER:   mpiexec -n 4 Parte_3
// ----------------------

// Meter os includes das bibliotecas
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Definição de constantes para que antes de compilar, o ficheiro de codigo que vai ser compilado, substitua todas as instanias pelo que esta à frente
#define TOL 1e-6 // tolerancia
#define ITERMAX 500000 // criterio de paragem
#define NXMAX 500 // para evitar alocação de memoria excessiva. definir numero maximo de pontos da grelha
#define L 1.0 // Tamanho do lado do quadrado, do dominio

// Definir uma função qualquer
double f(double x, double y){
    return (x + y);
}

int main(int argc, char *argv[]) {

    // Declarar variaveis locais
    int nprocs; // Numero de processos
    int myid; // Rank  de cada processo
    int nx, ny; // Tamanho da matriz para ambas as coordenadas

    // Definir o MPI
    MPI_Init(&argc, &argv); // Inicializar o MPI -> A partir daqui, cada processo vai correr "individualmente"
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Definir o tamanho do comunicador MPI. Aqui fornecemos o endereço para onde esta armazenado o valor da variável n_procs
    MPI_Comm_rank(MPI_COMM_WORLD, &myid); // Definir o rank de cada processo. O rank tem siginifcado apenas no contexto do comunicador

    // Isolar o processo para fazer o scanf para eprguntar apenas uma vez e nao em todos os processos, o tamanho que utilizador quer para a matriz
    if(myid == 0){ // Isolamos o processo 0

        // Printar para pedir ao utilizador
        printf("Introduza o número de pontos (max %d, 0 para sair): ", NXMAX);
        // Ir buscar o valor
        scanf(" %d", &nx); // Guardamos em nx. Nota que nx = ny

    }

    // Fazer o Brodcast para que todos os processos tenham a informação adequirida anteriomente
    // Todos os processo que tem id diferente de 0 recebem. O que tiver igual a 0 envia o nx. Um comunica com todos
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD); // O que queremos enviar/receber, o tamnho, o tipo de dados, quem esta a enviar
    
    // Definir o ny
    ny = nx;

    // Vamos testar o valor nx, introduzido pelo utilizador
    if(nx == 0){ // O utilizador quer sair

        // Vamos finalizar o MPI antes de sair
        MPI_Finalize();
        // Sair
        return 0;

    }
    else if (nx > NXMAX) // Algo inválido
    {
        // Vamos finalizar o MPI antes de sair
        MPI_Finalize();
        // Sair
        return 1;
    }
    
    // Definição e inicialização de variáveis para a criação do comunicador cartesiano
    int ndims = 2; // Numero de dimensoes
    int dims[2] = {(int)(nprocs / 2), 2}; // NUmero de linhas, numero de colunas
    int periodic[2] = {0, 0}; // É uma variável booleana, so que em C nao há, entao metemos um int. Mas so pode ser 0 ou 1
    MPI_Comm comm2D;
    // Vamos criar um comunicador cartesiano. NOTA: um comunicador que é criado apartir de outro nao pode ter mais processos do que o seu criador, mas pode ter menos
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periodic, 1, &comm2D); // comunicador anterior, numero de dimensoes, tamnho do comunicador, periodicidade, reorder, datatype do comunicador
    
    // Definir o novo id, isto é, o novo rank
    int newid;
    // Vamos ver quais sao os novos ranks, dai temos criado um novo comunicador
    MPI_Comm_rank(comm2D, &newid);

    // Definir os vizinhos
    int nbrbottom, nbrtop, nbrright, nbrleft;
    // Cada processo tem de saber o rank dos seus vizinhos, o de cima e o de baixo
    MPI_Cart_shift(comm2D, 0, 1, &nbrtop, &nbrbottom); // Fazemos o shift no comm2D para descobrir vizinho de cima e baixo
    MPI_Cart_shift(comm2D, 1, 1, &nbrleft, &nbrright); // Fazemos o shift no comm2D para descobrir vizinho da direita e da esquerda

    // Printar para ver/dar debug as direções do shift; uma verificação
    printf("newid = %d   nbrtop = %d     nbrbottom = %d     nbrleft = %d    nbrright = %d\n", newid, nbrtop, nbrbottom, nbrleft, nbrright);
    MPI_Barrier(comm2D); // Para organizar os prints. VErifica se todos os processos ja chegaram aqui. SE ainda falta algum, esperam. Serve para sincronizar os processos
                        // O primeiro a acabar fica a espera do ultimo
    // Declarar variaveis que vao ser utilziadas por todos os processos
    int firstrow, nrows;
    int firstcol, ncols; // Para a versao 2D

    // Identificar qual o processo que nao é incluido no novo comunicador. Este nao vai guardar em comm2D poque nao tem rank nem acesso. Entao é devolvido o apotador null que permite descobrir qual o processo
    if(comm2D == MPI_COMM_NULL){

        // Terminamos o processo
        MPI_Finalize();
        // E terminamos/fechamos o ambeinte MPI neste proesso
        return 0;

    }

    // Definir isto para nao estar sempre a aceder a um array, so uma questao de efeciencia
    int dim0 = dims[0];
    // Atualizar o numero de processos
    nprocs = dim0 * dims[1];

    // O processo 0, porque este existe sempre, é quem cria estes arrays
    if(newid == 0){
        
        // Fazer a distribuição das linhas e colunas
        int listfirstrow[nprocs];
        int listnrows[nprocs];

        int listfirstcol[nprocs];
        int listncols[nprocs];

        // Numero de linhas por processo, dividimos pelo numero de factias em vez do numero de processos
        int nrowsP = (int)(((double)(ny - 2)) / ((double)dim0) + 0.5);

        // Ciclo for para percorrer todos os processos. O i é o rank do processo
        for(int i = 0; i < dim0 - 1; i++){
            
            // Para cada processo indicamos
            listfirstrow[2 * i] = 1 + i * nrowsP; 
            listnrows[2 * i] = nrowsP; 

            listfirstrow[2 * i + 1] = 1 + i * nrowsP;
            listnrows[2 * i + 1] = nrowsP;

        }

        // Linhas
        listfirstrow[nprocs - 1] = 1 + (dim0 - 1) * nrowsP; 
        listnrows[nprocs - 1] = ny - 2 - (dim0 - 1) * nrowsP;

        listfirstrow[nprocs - 2] = 1 + (dim0 - 1) * nrowsP;  
        listnrows[nprocs - 2] = ny - 2 - (dim0 - 1) * nrowsP;

        // Colunas
        for(int i = 0; i < dim0; i++){

            // Coluna dos pares
            listfirstcol[2 * i] = 1; // São os processo 0, 2, 4, 6, ... Ver esquema no caderno
            listncols[2 * i] = (int)(nx / 2) - 1; // -1 porque a primeira nao conta
            // Coluna dos ímpares
            listfirstcol[2 * i + 1] = (int)(nx / 2); // Metade dos pontos de uma linha
            listncols[2 * i + 1] = nx - 1 - (int)(nx / 2); 

        }

        // Linhas
        MPI_Scatter(listfirstrow, 1, MPI_INT, &firstrow, 1, MPI_INT, 0, comm2D);
        MPI_Scatter(listnrows, 1, MPI_INT, &nrows, 1, MPI_INT, 0, comm2D);
        // Colunas
        MPI_Scatter(listfirstcol, 1, MPI_INT, &firstcol, 1, MPI_INT, 0, comm2D);
        MPI_Scatter(listncols, 1, MPI_INT, &ncols, 1, MPI_INT, 0, comm2D);

    }
    // Os restantes recebem o array que foi criado
    else{

        // Aqui so se vai receber, entao metemos um apontador NULL (MPI_BOTTOM) que nao precisamos de enviar nada. É sempre o 0 que envia. Ele esta aqui para os outros processos saberem que nao sao eles que enviam. Eles tem o papel de receber
        MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &firstrow, 1, MPI_INT, 0, comm2D);
        MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &nrows, 1, MPI_INT, 0, comm2D);

        MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &firstcol, 1, MPI_INT, 0, comm2D);
        MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &ncols, 1, MPI_INT, 0, comm2D);

    }

    // Verificar se a atribuição das linhas e colunas foi bem feita. A last row de um, tem de ser igual a firstrow do proximo + 1
    printf("newid = %d   firstrow = %d     lastrow = %d     firstcol = %d       lastcol = %d\n", newid, firstrow, firstrow + nrows - 1, firstcol, firstcol + ncols - 1);
    MPI_Barrier(comm2D);

    // Declarar matrizes com declaração dinamica
    double (*Vnew)[ncols + 2]; // Número de linhas da matriz Vnem. O +2 e por causa das fantasma
    double (*Vold)[ncols + 2];
    double (*myf)[ncols + 2];
    Vnew = calloc(nrows + 2, sizeof(*Vnew)); // O numero de linhas tem de ser igual ao numero de linhas +2, as tais "fantastma". Tamanho de cada elemento
                                             // Para aceder a esta matriz fazemos Vnem[i][j]
                                             // calloc inicializa a memoria alocada a 0. O malloc apenas aloca a memoria
    Vold = calloc(nrows + 2, sizeof(*Vold));
    myf = calloc(nrows + 2, sizeof(*myf));

    // Definir o h
    double h = L / ((double)(nx - 1)); // É a distancia entre pontos consecutivos. É o numero de intervalors

    // Inicializar a matriz da função f, ou seja, meter la cenas
    for(int i = 1; i <= nrows; i++){
        for(int j = 1; j <= ncols; j++ ){
            // Preencher myf
            myf[i][j] = f((-L/2.0) + (firstcol + j - 1) * h, (L/2.0) - (firstrow + i - 1) * h); // Para 2D
        }

    }

    // Preencher as condições fronteira. Esses valores nao sao alterados mas necessitam de ser inicializados
    // Para o processo de cima/linha de cima que agora esta no processo 0 e 1
    if(newid == 0 || newid == 1){
        // Percorrer elementos da linha de cima
        for(int i = 0; i < (ncols + 2); i++){ // Ou começar em i = 1 e ter i < ncols
            // Preeenche o valor da condição fronteira, elemento a elemento
            Vnew[0][i] = 0;
            Vold[0][i] = 0;
        }
    }
    // Para o processo de baixo/linha de baixo
    if(newid == (nprocs - 1) || newid == (nprocs - 2)){
        // Percorrer elementos da linha de baixo
        for(int i = 0; i < (ncols + 2); i++){
            // Preeenche o valor da condição fronteira, elemento a elemento
            Vnew[nrows + 1][i] = 0;
            Vold[nrows + 1][i] = 0;
        }
    }

    // Se der 0 é par, se der 1 é ímpar. Fazemos isto porque as condições fronteira sao diferentes para os processo par e ímpar
    if(newid % 2 == 0){

        // ...
        for(int i = 0; i < (nrows + 2); i++){
            // Preeenche o valor da condição fronteira da esquerda, elemento a elemento
            Vnew[i][0] = 0;
            Vold[i][0] = 0;
        }

    }
    else{
        
        // ...
        for(int i = 0; i < (nrows + 2); i++){
            // Preeenche o valor da condição fronteira da direita, elemento a elemento
            Vnew[i][ncols + 1] = 0;
            Vold[i][ncols + 1] = 0;
        }

    }

    // Criar o data type para as colunas
    MPI_Datatype column;
    // Definir
    MPI_Type_vector((nrows + 2), 1, (ncols + 2), MPI_DOUBLE, &column); // Numero total blocos/colunas, numero de elementos, periodicidade, O tipo dos elementos, oonde armazenar
    // Dar commit
    MPI_Type_commit(&column);

    // COmeçar contador de EScrita
    double tm1 = MPI_Wtime();

    // Iterar por cada dominio - Iterar o momento Jacobi
    for(int iter = 0; iter < ITERMAX; iter++){

        // Definir um array para somas
        double sums[2];

        // Linhas do subdominio
        for(int i = 1; i <= nrows; i++){

            // Colunas do subdminio
            for(int j = 1; j <= ncols - 1; j++){

                // Preencher a matriz Vnew
                Vnew[i][j] = (Vold[i+1][j] + Vold[i-1][j] + Vold[i][j-1] + Vold[i][j+1] - h*h*myf[i][j]) / 4.0;
                // Guardar a soma no array
                sums[0] += (Vnew[i][j] - Vold[i][j]) * (Vnew[i][j] - Vold[i][j]);
                sums[1] += Vnew[i][j] * Vnew[i][j];


            }

        }

        // Definir uma variável para guardar todas as somas
        double global_sums[2];
        // Meter as sums no global_sums
        MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, comm2D);

        // Vamos testar/comparar com a condição
        if((global_sums[0] / global_sums[1]) < TOL){

            // Ver o tempo de escrita
            if(newid == 0){
                printf("Calculo demorou %f segundos para %d iteraçoes\n", MPI_Wtime() - tm1, iter);
                tm1 = MPI_Wtime();
            }

            // Escrever o ficheiro -> NÃO VAMOS FAZER ISTO (AFINAL VAMOS)
            // Definir as variáveis necessárias
            int gsizes[2] = {ny, nx}; // Tamanho da matriz global
            int lsizes[2] = {nrows, ncols + 1}; // Tamanho da matriz local. O +1 é a coluna da fronteira vertical, uns à direita(rank ímpar) outros à esquerda(rank par)
            int start_ind[2] = {firstrow, firstcol - 1 + newid % 2}; // Índice de começo

            // Fazer ajustes ao tamanho -> Dois processo de cima
            if(newid == 0 || newid == 1){

                lsizes[0]++;
                start_ind[0]--;

            }
            // Fazer mais ajustes again -> Dois processos de baixo
            if(newid == nprocs - 2 || newid == nprocs - 1){

                lsizes[0]++;

            }

            // Definir um novo datatype para o ficheiro
            MPI_Datatype filetype;
            MPI_Type_create_subarray(2, gsizes, lsizes, start_ind, MPI_ORDER_C, MPI_DOUBLE, &filetype);
            MPI_Type_commit(&filetype);

            // Definir variáveis para a memoria
            int memsizes[2] = {nrows + 2, ncols + 2};
            start_ind[0] = 1;
            start_ind[1] = newid % 2;

            // AJuste aos dois processo de cima
            if(newid == 0 || newid == 1){

                start_ind[0] = 0;

            }

            // Definir um novo datatype para o memoria
            MPI_Datatype memtype;
            MPI_Type_create_subarray(2, memsizes, lsizes, start_ind, MPI_ORDER_C, MPI_DOUBLE, &memtype);
            MPI_Type_commit(&memtype);

            // Definir um ficheiro para efetuar escrita
            MPI_File fp;
            MPI_File_open(comm2D, "resultados_2D.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
            MPI_File_set_view(fp, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
            MPI_File_write_all(fp, Vnew, 1, memtype, MPI_STATUS_IGNORE);
            MPI_File_close(&fp);

            // Libertar memoria dos datatypes
            MPI_Type_free(&filetype);
            MPI_Type_free(&memtype);


            // Vamos dar print ao processo que esta em uso
            printf("\nID Processo: %d\n", newid);
            // Vamos dar print ao que o processo fez
            for(int i = 0; i < nrows + 2; i++){
                for(int j = 0; j < (ncols + 2); j++){
                    // Elemento da matriz
                    printf("%f  ", Vnew[i][j]);
                }
                // Introduzir uma mundaça de linha
                printf("\n");
            }


            // Terminar depois da escrita
            break;

        }

        // Antes de passar à próxima iteração temos de trocar as linhas fantasma
        MPI_Sendrecv(Vnew[1], (ncols + 2), MPI_DOUBLE, nbrtop, 0, Vnew[nrows + 1], (ncols + 2), MPI_DOUBLE, nbrbottom, 0, comm2D, MPI_STATUS_IGNORE); // Sentido ascendente
        MPI_Sendrecv(Vnew[nrows], (ncols + 2), MPI_DOUBLE, nbrbottom, 1, Vnew[0], (ncols + 2), MPI_DOUBLE, nbrtop, 1, comm2D, MPI_STATUS_IGNORE); // Sentido descendente
        
        MPI_Sendrecv(&(Vnew[0][ncols]), 1, column, nbrright, 2, &(Vnew[0][0]), 1, column, nbrleft, 2, comm2D, MPI_STATUS_IGNORE); // Sentido para a direita
        MPI_Sendrecv(&(Vnew[0][1]), 1, column, nbrleft, 3, &(Vnew[0][ncols + 1]), 1, column, nbrright, 3, comm2D, MPI_STATUS_IGNORE); // Sentido para a esquerda

        // Atualizar o array, isto é, dizer que o Vold da proxima iteração é o Vnew que acabamos de calcular. Tem de ser feito elemento a elemento: Vold[][] = Vnew[][]
        // Ou seja, copiar o Vnew para o Vold
        for(int i = 0; i < nrows + 2; i++){
            for(int j = 0; j < (ncols + 2); j++){
                Vold[i][j] = Vnew[i][j];
            }
        }

    }

    // Dar free ao data type
    MPI_Type_free(&column);

    // Libertar memoria
    free(Vnew);
    free(Vold);
    free(myf);

    // Finalizar o MPI
    MPI_Finalize();
    return 0;

}