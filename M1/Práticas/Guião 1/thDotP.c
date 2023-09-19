// Secção dos Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// Função POSIX das Threads
/* 
int pthread_create(pthread_t *thread, const pthread_attr_t *attr, 
                      void *(*start_routine) (void *), void *arg); 
*/

// Definir o número de threads apriori pois so vamos usar 2
#define NUM_THREADS 2

// Definição de uma estrutura para passar para a função 'dot_product'
struct vecs_info{
    
    // Tamanho máximo que o índice vai percorrer
    int n;
    // Indices para cada thread. Basta um pois vao operar nos mesmos indices
    int indx_thread;
    // Ponteiros para os vetores vec_1 e vec_2
    float* v1;
    float* v2;

};


// Função para calcular o produto de vetores/arrays de floats
void* dot_product(void *vecs){

    // Variável que armazena o resultado
    float result = 0.0;
    // Tranformar o argumento numa estrutura
    struct vecs_info v_i = *(struct vecs_info*) vecs;
    // Definir as variáveis da estrutura (NÃO ESTA A CHEGAR AQUI)
    int n_max = v_i.n;
    int idx_t = v_i.indx_thread;
    float* vec_1_ref = v_i.v1;
    float* vec_2_ref = v_i.v2;

    // Ciclo
    for(int i = idx_t; i < n_max; i++){

        // Fazer a operação de dot product
        result += vec_1_ref[i] * vec_2_ref[i];

    }
    printf("O resultado do 'dot product' = %f\n", result);

    // Vamos alocar espaço para guardar o resultado
    float* resulatado_thread = (float *)malloc(sizeof(float));
    // Guardar o resultado noo endereço
    *resulatado_thread = result;

    // Vamos retornar o resultado
    pthread_exit(resulatado_thread);
}

int main (int argc, char *argv[]){

    // Condição que verifica numero de argumentos
    if(argc != 1){

        printf("Não introduza nenhum argumento.\n");
        exit(1);

    }

    // Definição do array dos ids das threads
    pthread_t threads_id[NUM_THREADS];
    // Definição da resposta da criação das threads
    int rc;
    // Definir as estruturas
    struct vecs_info idx_thr[NUM_THREADS];
    // Definição dos dois vetores
    float vec_1[2000];
    float vec_2[2000];
    // Definir os ponteiros para os vetores da estrutura
    idx_thr->v1 = (float *)vec_1;
    idx_thr->v2 = (float *)vec_2;

    // Meter valores nos vetores
    printf("Vec_1\t\tVec_2\n\n");
    for(int i = 0; i < 2000; i++){

        // Gerar um valor aleatorio para o vec_1
        float r1 = (float)rand()/(float)(RAND_MAX/2000);
        printf("%f\t",r1);
        float r2 = (float)rand()/(float)(RAND_MAX/2000);
        printf("%f\n",r2);
        // Meter os valores dentro dos vetores
        vec_1[i] = r1;
        vec_2[i] = r2;

    }
    printf("\n\n");

    // Ciclo para criar as threads
    for(int i = 0; i < NUM_THREADS; i++){
        idx_thr[i].v1 = (float *)vec_1;
        idx_thr[i].v2 = (float *)vec_2;

        // Criar as threads
        if(i == 0){
            // Definir os parametros especificos para cada thread
            idx_thr[i].n = 1000;
            idx_thr[i].indx_thread = 0;
            //Criar a thread
            rc = pthread_create(&threads_id[i], NULL, dot_product, (void *) &idx_thr[i]);
            // Print que indica quando a thread é criada
            printf("main(): thread created. Number: %d.\n", i);
        }
        else{
            // Definir os parametros especificos para cada thread
            idx_thr[i].n = 2000;
            idx_thr[i].indx_thread = 1000;
            //Criar a thread
            rc = pthread_create(&threads_id[i], NULL, dot_product, (void *) &idx_thr[i]);
            // Print que indica quando a thread é criada
            printf("main(): thread created. Number: %d.\n", i);
        }

    }

    // Conta final
    float soma_final = 0.0;
    // Definir a variável para it buscar o resultado
    float* thread_result;
    // Ciclo for que assegura a thread principal espera que todas as outras acabem
    for(int i = 0; i < NUM_THREADS; i++){

        // Função que se encarrega de assegurar que as threads esperam que as outras terminem
        // Passamos o endereço do ponteiro para que seja la escrito o resutlado retornado da thread
        pthread_join(threads_id[i], &thread_result);

        // Fazer a conta final
        soma_final += *thread_result;
        // Libertamos a memoria
        free(thread_result);

    }

    // Verificar a soma dos vetores
    float resultado_final = 0.0;
    for(int i = 0; i < 2000; i++){

        resultado_final += vec_1[i] * vec_2[i];

    }

    printf("\n\n");
    printf("Resultado final esperado: %f", resultado_final);
    printf("\n\n");
    // Printar o resultado final obtido
    printf("Resultado final obtido: %f", soma_final);
    printf("\n");

    pthread_exit(NULL);
}
