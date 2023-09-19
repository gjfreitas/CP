// Projeto 2
// Gonçalo Freitas 98012

// Para correr
// mpicc 98012_a.c -o 98012_a -lm
// mpiexec -n 4 ./98012_a


// d)
// O  método  de  relaxação  de  Gauss-Seidel  é  uma  alternativa  ao  método  de  Jacobi,  que 
// converge mais rapidamente para a solução.  Trata-se de numa modificação do método de Jacobi 
// que consiste na utilização dos valores mais recentes que estiverem disponíveis em cada momento. 
// Ou seja, para os vizinhos cujo 𝑉(𝑘) ainda não tenha sido calculado na iteração 𝑘 usa-se 𝑉(𝑘−1) tal 
// como no método de Jacobi, mas quando o 𝑉(𝑘) do vizinho já foi calculado na iteração atual usa-se 
// esse o valor mais recente.  
// Num programa sequencial (não paralelizado) bastaria substituir as linhas  
 
// for (i=1; i<=myrows; i++) { 
// for (j=1; j<=mycols; j++) { 
// Vnew[i][j] = (Vold[i-1][j] + Vold[i+1][j]  
// + Vold[i][j-1] + Vold[i][j+1] - h*h*myf[i][j])/4.0; 
 
// pelas linhas 
 
// for (i=1; i<=myrows; i++) { 
// for (j=1; j<=mycols; j++) { 
// Vnew[i][j] = (Vnew[i-1][j] + Vnew[i+1][j]  
// + Vnew[i][j-1] + Vnew[i][j+1] - h*h*myf[i][j])/4.0; 
 
// para  obter  o  método  de  Gauss-Seidel.  Esta  alteração  ao  programa  sequencial  corresponde  à 
// implementação  a  equação  iterativa  𝑉𝑖,𝑗(𝑘) =1
// 4[𝑉𝑖−1,𝑗(𝑘) +𝑉𝑖,𝑗−1(𝑘) +𝑉𝑖+1,𝑗(𝑘−1) +𝑉𝑖,𝑗+1(𝑘−1) −
// ℎ2𝑓𝑖,𝑗].  Porém,  num  programa  paralelizado  não  seria  possível  usar  esta  equação  para  todos  os 
// pontos. Explique no relatório qual seria o problema. 
// A  estratégia  mais  simples  para  superar  esse  problema  é  a  utilização  de  um  esquema  de 
// atualização do tipo vermelho-preto (ou par-ímpar). Resumidamente, neste esquema cada 
// processo executa os seguintes passos numa iteração (incluindo duas fases distintas de 
// comunicação): 
// (i) Inicialmente calcula os  𝑉𝑖,𝑗(𝑘) de todos os pontos para os quais 𝑖+𝑗 é par, usando os   
// valores  de  𝑉(𝑘−1)  dos  4  pontos  vizinhos  na  iteração  𝑘−1.  (Note  que  os  4  pontos 
// vizinhos de um ponto ‘par’ são todos ‘ímpares’, e vice-versa. Note também que para 
// determinar corretamente a paridade de um ponto, devem de ser usados os índices 𝑖 e 
// 𝑗 correspondentes à matriz global.);  
// (ii) Comunica aos processos vizinhos os valores atualizados dos pontos ‘pares’ na fronteira 
// do subdomínio; 
// (iii) Calcula os valores 𝑉𝑖,𝑗(𝑘) com 𝑖+𝑗 ímpar, usando os 4 valores dos pontos vizinhos que 
// foram atualizados no passo (i) (e que são todos ‘pares’); 
// (iv) Finalmente, comunica os pontos ‘ímpares’ aos processos vizinhos, para serem usados 
// na próxima iteração.  
 
// Partindo do programa da alínea b), converta o método de Jacobi no  método de Gauss-Seidel 
// aplicando  este  esquema  de  atualização.  Tal  como  na  alínea  b),  use  o  estêncil  de  5  pontos  com 
// condições fronteira periódicas. Se não resolveu a alínea b) (apenas nesse caso) use as condições 
// fronteira da alínea a). Deve pesquisar os pormenores este algoritmo na bibliografia. Sugere-se que 
// comece pela leitura do capítulo 2, secção 2.2, do livro de Jianping Zhu, Solving Partial Differential 
// Equations on Parallel Computers, World Scientific, 1994.

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TOL 1e-6 // do encunciado
#define ITERMAX 500000
#define NXMAX 500
#define L 1.0

double f(double x, double y){
   // 𝑓(𝑥, 𝑦) = 7 sin(2𝜋𝑥) cos(3𝜋𝑥) sin(2𝜋𝑦) cos(3𝜋𝑦)
    return 7*sin(2*M_PI*x)*cos(3*M_PI*x)*sin(2*M_PI*y)*cos(3*M_PI*y);
}

int main(int argc, char *argv[])
{
    int nprocs;
    int myid; 
    int nx, ny;

    // manager_rank = 0

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0)
    {
        printf("Introduza numero de pontos {max %d, 0 para sair}: ",NXMAX);
        scanf(" %d", &nx);
        //nx = 100; // para testar
    }

    // 0 envia, os outros recebem
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ny = nx;

    // verificar se nx esta dentro dos limites
    if (nx == 0)
    {
        MPI_Finalize();
        return 0;
    }

    if (nx < 0 || nx > NXMAX)
    {
        MPI_Finalize();
        return 1;
    }

    int nprocs_col = (int) nprocs/2;

    // Definição e inicialização de variáveis para a criação do comunicador cartesiano
    int ndims = 2;
    int dims[2] = {nprocs_col, 2};
    int periodic[2] = {1,1};
    MPI_Comm comm2D;

    // Criar um comunicador cartesiano
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periodic, 1, &comm2D);

    // Definir o novo rank
    int newid;
    MPI_Comm_rank(comm2D, &newid);

    // Definir os vizinhos
    int nbrbottom, nbrtop, nbrleft, nbrright;

    // Identificar qual o processo que nao é incluido no novo comunicador
    // Este nao vai guardar em comm2D poque nao tem rank nem acesso. 
    // Entao é devolvido o apotador null que permite descobrir qual o processo
    if (comm2D == MPI_COMM_NULL)
    {
        MPI_Finalize();
        return 0;
    }

    // Atualizar o numero de processos
    nprocs = nprocs_col * dims[1];

    // Cada processo tem de saber o rank dos seus vizinhos
    MPI_Cart_shift(comm2D, 0, 1, &nbrbottom , &nbrtop); // vizinhos de cima e de baixo
    MPI_Cart_shift(comm2D, 1, 1, &nbrleft , &nbrright); // vizinhos da esquerda e da direita

    printf("myid=%d, newid=%d, bot=%d, top=%d, left=%d, right=%d\n", myid, newid, nbrbottom, nbrtop, nbrleft, nbrright);   

    // Variáveis para o Jacobi
    int firstrow, firstcol;
    int myrows, mycols;

    if (newid == 0){
        // Fazer a distribuição das linhas e colunas
        int listfirstrow[nprocs];
        int listmyrows[nprocs];

        int listfirstcol[nprocs];
        int listmycols[nprocs];

        // Numero de linhas por processo
        int nrows = (int)((double)(ny-2)/(double)nprocs_col + 0.5);
        printf("nprocs=%d, nrows=%d\n", nprocs, nrows);  

        // Linhas
        for (int i = 0; i < nprocs_col; i++)
        {
            listfirstrow[2*i] = i *  (nrows);
            listmyrows[2*i] = nrows+1;
            listfirstrow[2*i+1] = i *  (nrows);
            listmyrows[2*i+1] = nrows+1;
        }

        // Altera o numero de linhas do penultimo e do ultimo
        listfirstrow[nprocs-2] = 1 + (nprocs_col-1)*nrows;
        listfirstrow[nprocs-1] = 1 + (nprocs_col-1)*nrows;
        listmyrows[nprocs-2] = ny - (nprocs_col - 1) * nrows;
        listmyrows[nprocs-1] = ny - (nprocs_col - 1) * nrows;


        // Colunas
        int ncols_temp = (int)((nx-2)/2);
        for (int i = 0; i < nprocs_col; i++)
        {
            listfirstcol[2*i] = 0;
            listmycols[2*i] = ncols_temp + 1;
            listfirstcol[2*i+1] = ncols_temp + 1;
            listmycols[2*i+1] = nx - 1 - ncols_temp;
        }

        // Linhas
        MPI_Scatter(listfirstrow, 1, MPI_INT, &firstrow, 1, MPI_INT, newid, comm2D);
        MPI_Scatter(listmyrows, 1, MPI_INT, &myrows, 1, MPI_INT, newid, comm2D);
        // Colunas
        MPI_Scatter(listfirstcol, 1, MPI_INT, &firstcol, 1, MPI_INT, newid, comm2D);
        MPI_Scatter(listmycols, 1, MPI_INT, &mycols, 1, MPI_INT, newid, comm2D);;
    }
    else
    { // Os restantes recebem o array que foi criado

        // Só receber e não enviar -> apontador NULL (MPI_BOTTOM) 
        MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &firstrow, 1, MPI_INT, 0, comm2D);
        MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &myrows, 1, MPI_INT, 0, comm2D);

        MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &firstcol, 1, MPI_INT, 0, comm2D);
        MPI_Scatter(MPI_BOTTOM, 1, MPI_INT, &mycols, 1, MPI_INT, 0, comm2D);
    }

    // Verificar se a atribuição das linhas e colunas foi bem feita
    printf("newid = %d   firstrow = %d     lastrow = %d     firstcol = %d       lastcol = %d\n", newid, firstrow, firstrow + myrows - 1, firstcol, firstcol + mycols - 1);
    MPI_Barrier(comm2D);

    // Alocar memória para os arrays
    double (*Vold)[mycols+2], (*Vnew)[mycols+2], (*myf)[mycols+2];
    Vold = calloc(myrows + 2, sizeof(*Vold));
    Vnew = calloc(myrows + 2, sizeof(*Vnew));
    myf = calloc(myrows + 2, sizeof(*myf));

    // h é a distancia entre pontos consecutivos (Nº de intervalors)
    double h = ((double)2 * L) / ((double) nx);

    // Inicializar a matriz da função f (função f definida no inicio consoante o enunciado)
    for (int j = 1; j < mycols + 1 ; j++)
    {
        for (int i = 1; i < myrows + 1; i++)
        {
            myf[i][j] = f(-L + (firstcol + j - 1) * h, -L + (firstrow + i - 1) * h);
        }
        
    }

    // !!! OS CICLOS PARA DEFINIR OS VALORES NAS FRONTEIRAS FORAM REMOVIDOS (alinea b)

    // Criar o data type para as colunas
    MPI_Datatype column;
    MPI_Type_vector(myrows + 2, 1, mycols + 2 , MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    // Contador de Escrita
    double tm1 = MPI_Wtime();

    // Jacobi
    for (int iter = 0; iter < ITERMAX; iter++)
    {   
        // Definir um array para somas
        double sums[2] = {0.0,0.0};
        // Definir uma variável para guardar todas as somas
        double global_sums[2];


        // !!! ALTERADO CALCULOS PARES (i+j é par)
        for (int i = 1; i < myrows + 1; i++)
        {
            for (int j = 1; j < mycols + 1 ; j++)
            {
                if (((firstcol + j - 1)  + (firstrow + i - 1)) %2 == 0) // VERIFICA QUE É PAR
                {
                    Vnew[i][j] = (Vnew[i+1][j] + Vnew[i-1][j] + Vnew[i][j+1] + Vnew[i][j-1]  - h * h  * myf[i][j]) / 4.0;
                    sums[0] += (Vnew[i][j] - Vold[i][j]) * (Vnew[i][j] - Vold[i][j]);
                    sums[1] += Vnew[i][j] * Vnew[i][j];
                }
            }  
        }

        // !!! ALTERADO COMUNICAÇÕES AOS VIZINHOS (pares para ímpares)
        MPI_Sendrecv(&Vnew[1][1], mycols, MPI_DOUBLE, nbrbottom, 4, &Vnew[myrows+1][1], mycols, MPI_DOUBLE, nbrtop, 4, comm2D, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&Vnew[myrows][1], mycols, MPI_DOUBLE, nbrtop, 5, &Vnew[0][1], mycols, MPI_DOUBLE, nbrbottom, 5, comm2D, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&Vnew[1][1], 1, column, nbrleft, 6, &Vnew[1][mycols+1], 1, column, nbrright, 6, comm2D, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&Vnew[1][mycols], 1, column, nbrright, 7, &Vnew[1][0], 1, column, nbrleft, 7, comm2D, MPI_STATUS_IGNORE);

        // !!! ALTERADO CALCULOS ÍMPARES (i+j é ímpar)
        for (int i = 1; i < myrows + 1; i++)
        {
            for (int j = 1; j < mycols + 1 ; j++)
            {
                if (((firstcol + j - 1)  + (firstrow + i - 1)) %2 == 1) // verifca que é impar
                {
                    Vnew[i][j] = (Vnew[i-1][j] + Vnew[i][j-1] + Vnew[i][j+1] + Vnew[i+1][j] - h * h * myf[i][j]) / 4.0 ;
                    sums[0] += (Vnew[i][j]-Vold[i][j])*(Vnew[i][j]-Vold[i][j]);
                    sums[1] += Vnew[i][j]*Vnew[i][j];
                }
            }
        }

        // Comunicar aos vizinhos (ímpares para pares)
        // !!! ALTERADO COMUNICAÇÕES AOS VIZINHOS (ímpares para pares)
        MPI_Sendrecv(&Vnew[1][1], mycols, MPI_DOUBLE, nbrbottom, 8, &Vnew[myrows+1][1], mycols, MPI_DOUBLE, nbrtop, 8, comm2D, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&Vnew[myrows][1], mycols, MPI_DOUBLE, nbrtop, 9, &Vnew[0][1], mycols, MPI_DOUBLE, nbrbottom, 9, comm2D, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&Vnew[1][1], 1, column, nbrleft, 10, &Vnew[1][mycols+1], 1, column, nbrright, 10, comm2D, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&Vnew[1][mycols], 1, column, nbrright, 11, &Vnew[1][0], 1, column, nbrleft, 11, comm2D, MPI_STATUS_IGNORE);

        // Guardar as sums no global_sums
        MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, comm2D); // MPI_SUM 
        
        if (sqrt(global_sums[0]/global_sums[1]) < TOL)
        //if (global_sums[0]/global_sums[1] < TOL)
        {   
            // Ver o tempo de escrita
            if (newid == 0)
            {
                printf("Calculo demorou %f seg; %d iteracoes\n", MPI_Wtime()-tm1, iter);
                tm1 = MPI_Wtime();
            }

            // !!! DESTA FORMA, NÃO É NECESSÁRIO TER EM CONTAS AS COLUNAS FANTASMA (alinea b)
            // !!! ISTO TORNA O CÁLCULO DE lsizes E start_ind MAIS FÁCIL (NÃO É PRECISO VERIFICAÇÕES EXTRA) (alinea b)
            int gsizes[2] = {ny, nx}; // Tamanho da matriz global
            int lsizes[2] = {myrows, mycols}; ; // Tamanho da matriz local. O +1 é a coluna da fronteira vertical, uns à direita(rank ímpar) outros à esquerda(rank par)
            int start_ind[2] = {firstrow, firstcol}; // Índice de começo

            // !!! AS VERIFICAÇÕES FORAM ENTÃO REMOVIDAS (alinea b)

            // Definir um novo datatype para o ficheiro
            MPI_Datatype filetype;
            MPI_Type_create_subarray(2, gsizes, lsizes, start_ind, MPI_ORDER_C, MPI_DOUBLE, &filetype);
            MPI_Type_commit(&filetype);
            
            // Definir variáveis para a memoria
            int memsizes[2] = {myrows+2, mycols+2};
            start_ind[0] = 1;
            start_ind[1] = newid % 2;

            // Ajuste aos dois processo de cima
            if (newid == 0 || newid == 1) {
                start_ind[0]--;
            }


            // Definir um novo datatype para o memoria
            MPI_Datatype memtype;
            MPI_Type_create_subarray(2, memsizes, lsizes, start_ind, MPI_ORDER_C, MPI_DOUBLE, &memtype);
            MPI_Type_commit(&memtype);

            // Definir um ficheiro para efetuar escrita
            MPI_File fp;
            MPI_File_open(comm2D, "results_d.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
            MPI_File_set_view(fp, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
            
            MPI_File_write_all(fp, Vnew, 1, memtype, MPI_STATUS_IGNORE);
            MPI_File_close(&fp);

            // Libertar memoria dos datatypes
            MPI_Type_free(&filetype);
            MPI_Type_free(&memtype);

            if (newid == 0)
            {
                printf("Escrita demorou %f seg\n", MPI_Wtime()-tm1);
            }

            /*
            // Print do processo que esta em uso
            printf("\nID Processo: %d\n", newid);
            // Print do que o processo fez
            for(int i = 0; i < nrows + 2; i++){
                for(int j = 0; j < (ncols + 2); j++){
                    // Elemento da matriz
                    printf("%f  ", Vnew[i][j]);
                }
                // Introduzir uma mundaça de linha
                printf("\n");
            }
            */
            break;
        }

        // comunicações sentido ascendente
        MPI_Sendrecv(Vnew[myrows], mycols+2, MPI_DOUBLE, nbrtop, 0, Vnew[0] , mycols+2, MPI_DOUBLE, nbrbottom, 0, comm2D, MPI_STATUS_IGNORE);
        
        // comunicações sentido descendente
        MPI_Sendrecv(Vnew[1], mycols+2, MPI_DOUBLE, nbrbottom, 1, Vnew[myrows+1] , mycols+2, MPI_DOUBLE, nbrtop, 1, comm2D, MPI_STATUS_IGNORE);
        
        // comunicações sentido para direita
        MPI_Sendrecv(&(Vnew[0][mycols]), 1, column, nbrright, 2, &(Vnew[0][0]), 1, column, nbrleft, 2, comm2D, MPI_STATUS_IGNORE);

        // comunicações sentido para esquerda
        MPI_Sendrecv(&(Vnew[0][1]), 1, column, nbrleft, 3, &(Vnew[0][mycols+1]), 1, column, nbrright, 3, comm2D, MPI_STATUS_IGNORE);
         
        // Atualizar o array Vold
        for (int i = 0; i < myrows + 2; i++)
        {
            for (int j = 0; j < mycols + 2; j++)
            {
                Vold[i][j] = Vnew[i][j];
            }
            
        }
        
    }

    // Dar free ao data type
    MPI_Type_free(&column);

    // Libertar memoria
    free(Vold);
    free(Vnew);
    free(myf);

    // Finalizar o MPI
    MPI_Finalize();

    return 0;
}

