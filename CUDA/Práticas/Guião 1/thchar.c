# include <stdio.h>
# include <pthread.h>
# include <stdlib.h>
# include <string.h>

# define NUM_THREADS 5


void *printch(void *pch) {
    char c = *(char *)pch;
    printf("%c\n", c);
    
	return NULL;
}

int main(int argc, char *argv[])
{
    pthread_t tids[1024];
	
	if ( argc != 2 ) {
		printf("Error: Invalid number of arguments");
	}

	int size = strlen(argv[1]);

	for(int i = 0; i < size; i++) {
		int rc = pthread_create(&tids[i], NULL, printch, (void *)&argv[1][i]);
	}

	for(int i = 0; i < size; i++) {
		pthread_join(tids[i], NULL); // join é a primitiva que permite que uma thread espere pela outra
	}

	return 0;

    
}

