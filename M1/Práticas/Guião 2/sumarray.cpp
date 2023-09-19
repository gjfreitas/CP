
#include <stdio.h>
#include <time.h>

#define SIZE   (1000*4)
#define REPEAT (100000)

/**
 * sumarray using mmx instructions 
 * */
void sumarray_mmx( int *a, int *b, int *c, int size )
{

  for (int i=0;i<size;i+=2) {
    __asm__ volatile
        ( // instruction         comment          
        "\n\t movq     %1,%%mm0     \t#"
        "\n\t movq     %2,%%mm1     \t#"
        //"\n\t paddd    %%mm0,%%mm1    \t#" // default
        "\n\t paddusb    %%mm0,%%mm1    \t#"    // alinea 1.3 -> os resultados dao diferentes, falta 1 unidade
        "\n\t movq     %%mm1,%0     \t#"
        : "=m" (c[i])      // %0
        : "m"  (a[i]),     // %1 
          "m"  (b[i])      // %2
        );  
  }

   __asm__("emms" : : );
}

void sumarray_sse( int *a, int *b, int *c, int size )
{

  for (int i=0;i<size;i+=4) {
    __asm__ volatile
        ( // instruction         comment          
        "\n\t movdqa     %1,%%xmm0     \t#"
        "\n\t movdqa     %2,%%xmm1     \t#"
        "\n\t paddd    %%xmm0,%%xmm1    \t#" // default
        "\n\t movdqa     %%xmm1,%0     \t#"
        : "=m" (c[i])      // %0
        : "m"  (a[i]),     // %1 
          "m"  (b[i])      // %2
        );  
  }
}

/**
 * sumarray using classic code 
 * */
void sumarray( int *a, int *b, int *c, int size )
{
  for (int i=0;i<size;i++) {
      c[i]=a[i]+b[i];
  }
}

/**
 * print array
 * */
void print_array(int *a, int size)
{
    printf("base10: ");
    for (int i=0; i < size; i++) {
    printf("%10d",a[i]);
    }
    printf("\nbase16: ");
    for (int i=0; i < size; i++) {
    printf("%10x",a[i]);
    }
    printf("\n");
}

/**
 * init arrays
 * */
void initArrays( int *a, int *b, int *c, int size )
{
    for (int i=0; i< SIZE; i++) {
        a[i]=(i<<16)+1;
        b[i]=0xffff;
        c[i]=0;
    }
}


/**
 * test summation functions
 */
int main(void)
{
    int a[SIZE] __attribute__ ((aligned(16)));
    int b[SIZE] __attribute__ ((aligned(16)));
    int c[SIZE] __attribute__ ((aligned(16)));

    int n, nelemsum;

    clock_t init, end;

    //initialize arrays
    nelemsum=SIZE;
    initArrays(a,b,c,nelemsum);

    // test classic code
    init = clock();
    for(n=0;n<REPEAT;n++)
        sumarray(a,b,c,nelemsum);
    end = clock();

    print_array(c,12);

    printf("sumarray time = %f\n", (end-init)/(CLOCKS_PER_SEC*1.0));

    //initialize arrays
    initArrays(a,b,c,nelemsum);

    // test mmx code
    init = clock();
    for(n=0;n<REPEAT;n++)
        sumarray_mmx(a,b,c,nelemsum);
    end = clock();

    print_array(c,12);

    printf("sumarray time = %f\n", (end-init)/(CLOCKS_PER_SEC*1.0));

    printf("\n");

    // test sse code
    init = clock();
    for(n=0;n<REPEAT;n++)
        sumarray_sse(a,b,c,nelemsum);
    end = clock();

    print_array(c,12);

    printf("sumarray time = %f\n", (end-init)/(CLOCKS_PER_SEC*1.0));

    printf("\n");

    return 0;
}

    
