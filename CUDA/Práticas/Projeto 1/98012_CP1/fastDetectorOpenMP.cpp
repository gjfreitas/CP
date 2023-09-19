
// Based on CUDA SDK template from NVIDIA

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>
#include <sys/time.h>

// includes, project
#include <helper_cuda.h>
#include <helper_image.h>

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

#define MAX_BRIGHTNESS 255
#define FADEDIV 2

#define NUM_THREADS 16

// pixel base type
// Use int instead `unsigned char' so that we can
// store negative values.
typedef int pixel_t;

int offset[16]; // offsets to circle pixels

// initializes global array of offsets of neighborhood pixels in circle of radius 3
void makeOffsets(int offset[], const int row_stride)
{
    offset[0] = 0 + row_stride * 3;
    offset[1] = 1 + row_stride * 3;
    offset[2] = 2 + row_stride * 2;
    offset[3] = 3 + row_stride * 1;
    offset[4] = 3 + row_stride * 0;
    offset[5] = 3 + row_stride * -1;
    offset[6] = 2 + row_stride * -2;
    offset[7] = 1 + row_stride * -3;
    offset[8] = 0 + row_stride * -3;
    offset[9] = -1 + row_stride * -3;
    offset[10] = -2 + row_stride * -2;
    offset[11] = -3 + row_stride * -1;
    offset[12] = -3 + row_stride * 0;
    offset[13] = -3 + row_stride * 1;
    offset[14] = -2 + row_stride * 2;
    offset[15] = -1 + row_stride * 3;
}

#define DARKER (-1)
#define SIMILAR 0
#define BRIGHTER 1

// detects if pixel pointed to by h_ipixel is a FAST feature:
//   has at least th_count consecutive neighbours (along circle radius 3)
//   that are darker or brighter by th_diff
int fastCorner(const pixel_t *h_ipixel,
               const int w, const int h,
               const int th_count, // min count to detect corners size
               const int th_diff   // threshold diff to count
)
{
    int consec = 0;
    int dk_consec = 0, br_consec = 0;
    int dk_begin = 0, dk_begin_count = 0;
    int br_begin = 0, br_begin_count = 0;

    pixel_t pix_val = *h_ipixel;
    int intensity = SIMILAR;

    int p;
    for (p = 0; p < 16; p++)
    {
        if (h_ipixel[offset[p]] < pix_val - th_diff)
        { // Darker neighbor
            if (p == 0)
            {
                dk_begin = 1;
            }
            if (intensity == DARKER)
            {
                consec++;
            }
            else
            {
                if (intensity == BRIGHTER && consec > br_consec)
                {
                    if (br_begin == 1)
                    {
                        br_begin_count = consec;
                        br_begin = 0;
                    }
                    br_consec = consec;
                }
                consec = 1;
            }
            intensity = DARKER;
        }

        else if (h_ipixel[offset[p]] > pix_val + th_diff)
        { // Brighter neighbor
            if (p == 0)
            {
                br_begin = 1;
            }
            if (intensity == BRIGHTER)
            {
                consec++;
            }
            else
            {
                if (intensity == DARKER && consec > dk_consec)
                {
                    if (dk_begin == 1)
                    {
                        dk_begin_count = consec;
                        dk_begin = 0;
                    }
                    dk_consec = consec;
                }
                consec = 1;
            }
            intensity = BRIGHTER;
        }
        else
        { // Similar Neighbor
            if (intensity == DARKER && consec > dk_consec)
            {
                if (dk_begin == 1)
                {
                    dk_begin_count = consec;
                    dk_begin = 0;
                }
                dk_consec = consec;
            }
            if (intensity == BRIGHTER && consec > br_consec)
            {
                if (br_begin == 1)
                {
                    br_begin_count = consec;
                    br_begin = 0;
                }
                br_consec = consec;
            }
            consec = 0;
            intensity = SIMILAR;
        }
    }

    if (intensity == DARKER)
    {
        if (dk_begin_count)
        { // merge consecutive pixels
            if (consec + dk_begin_count > dk_consec)
                dk_consec = consec + dk_begin_count;
        }
        else if (consec > dk_consec)
        {
            dk_consec = consec;
        }
    }

    if (intensity == BRIGHTER)
    {
        if (br_begin_count)
        { // merge consecutive pixels
            if (consec + br_begin_count > br_consec)
                br_consec = consec + br_begin_count;
        }
        else if (consec > br_consec)
        {
            br_consec = consec;
        }
    }
    if (dk_consec >= th_count || br_consec >= th_count)
    {
        return 1;
    }
    return 0;
}

// returns the score of pixel pointed to by h_ipixel
int fastScore(const pixel_t *h_ipixel,
              const int w, const int h,
              const int th_count)
{
    int scoremin = 1;
    int scoremax = max(MAX_BRIGHTNESS-*h_ipixel,*h_ipixel);

    while (scoremax - scoremin > 1)
    {
        if (fastCorner(h_ipixel, w, h, th_count, (scoremin + scoremax) / 2))
        {
            scoremin = (scoremin + scoremax) / 2;
        }
        else {
            scoremax = (scoremin + scoremax) / 2;
        }
    }

    return scoremin;
}

// FAST non-maximum suppression
void nonMaximumSupression(const pixel_t *in, const pixel_t *corners,
                          pixel_t *nms,
                          const int w, const int h, const int th_count)
{
    int count = 0;

    int *corner_score = (int *) malloc(w*h * sizeof(int));

    // determine score of each corner
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            const int c = i * w + j;
            if (corners[c] == MAX_BRIGHTNESS)
            {
                  corner_score[c] = fastScore(in + c, w, h, th_count);
            }
            else {
                  corner_score[c] = 0;
            }
        }
    }

    // keep only corner with local maximum score
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            const int c = i * w + j;
            if (corners[c] != MAX_BRIGHTNESS)
            {
                continue;
            }

            int score_c = corner_score[c];
            int score_n;

            // check all neighbors
            for(int ni = max(0,i-1); ni < min(h,i+2); ni++) {
                for(int nj = max(0,j-1); nj < min(w,j+2); nj++) {
                    if(ni == i && nj == j) continue;

                    int nc = ni * w + nj; 
                    score_n = corner_score[nc];
                    if (score_n >= score_c) {
                        nms[c] = 0;
                        goto next;
                    }
                }
            }

            nms[c] = MAX_BRIGHTNESS;
            count++;
            next: continue;
        }
    }

    free(corner_score);

    printf("nonmax features %d\n", count);
}

// detects all FAST corners in image h_idata and marks them with MAX_BRIGHTNESS in h_odata
void fastDetectCorners(const pixel_t *h_idata,
                       const int w, const int h,
                       const int th_count, // min count to detect corners size
                       const int th_diff,  // threshold diff to count
                       pixel_t *h_odata)
{
    int i, j, count = 0;
    
    for (i = 3; i < h - 3; i++) // height image
    {
        for (j = 3; j < w - 3; j++) // width image
        {
            if (fastCorner(h_idata + i * w + j, w, h, th_count, th_diff))
            {
                h_odata[i * w + j] = MAX_BRIGHTNESS;
                count++;
            }
        }
    }
    printf("detected %d features\n", count);
}


// fast detector code to run on the host
void fastDetectorHost(const pixel_t *h_idata, const int w, const int h,
                      const int th_count, // min count to detect corners size
                      const int th_diff,  // threshold diff to count
                      const bool nonmaxflag,
                      pixel_t *h_odata)
{
    int i, j; // indexes in image

    // initialize h_odata to zero
    memset(h_odata, 0, h*w*sizeof(pixel_t));

    makeOffsets(offset, w);

    // corner detection
    fastDetectCorners(h_idata, w, h, th_count, th_diff, h_odata);


    if (nonmaxflag)
    {
        pixel_t *aux = (pixel_t *)malloc(w * h * sizeof(pixel_t));

        memcpy(aux, h_odata, w * h * sizeof(pixel_t));

        //non max supression
        nonMaximumSupression(h_idata, aux, h_odata, w, h, th_count);

        free(aux);
    }

    //add faded original image as background
    for (i = 0; i < h; i++) // height image
    {
        for (j = 0; j < w; j++) // width image
        {
            if(h_odata[i * w + j]!=MAX_BRIGHTNESS) {
                h_odata[i * w + j] = h_idata[i * w + j] / FADEDIV; // to obtain a faded background image
            }
        }
    }
}

// parallel version of nonMaximumSupression
void nonMaximumSupressionOPENMP(const pixel_t *in, const pixel_t *corners,
                          pixel_t *nms,
                          const int w, const int h, const int th_count)
{
    int count = 0;

    int *corner_score = (int *) malloc(w*h * sizeof(int));
    
    int i, j;

    // determine score of each corner
    #pragma omp parallel for num_threads(NUM_THREADS) private(i,j)
    for (i = 0; i < h; i++)
    {
        for (j = 0; j < w; j++)
        {
            const int c = i * w + j;
            if (corners[c] == MAX_BRIGHTNESS)
            {
                  corner_score[c] = fastScore(in + c, w, h, th_count);
            }
            else {
                  corner_score[c] = 0;
            }
        }
    }

    // keep only corner with local maximum score
    
    #pragma omp parallel for num_threads(NUM_THREADS) private(i,j) reduction(+:count)
    for (i = 0; i < h; i++)
    {
        for (j = 0; j < w; j++)
        {
            const int c = i * w + j;
            if (corners[c] != MAX_BRIGHTNESS)
            {
                continue;
            }

            int score_c = corner_score[c];
            int score_n;

            // check all neighbors
            for(int ni = max(0,i-1); ni < min(h,i+2); ni++) {
                for(int nj = max(0,j-1); nj < min(w,j+2); nj++) {
                    if(ni == i && nj == j) continue;

                    int nc = ni * w + nj; 
                    score_n = corner_score[nc];
                    if (score_n >= score_c) {
                        nms[c] = 0;
                        goto next;
                    }
                }
            }

            nms[c] = MAX_BRIGHTNESS;
            count++;
            next: continue;
        }
    }

    free(corner_score);

    printf("nonmax features %d\n", count);
}

// parallel version of fastDetectCorners
void fastDetectCornersOPENMP(const pixel_t *h_idata,
                       const int w, const int h,
                       const int th_count, // min count to detect corners size
                       const int th_diff,  // threshold diff to count
                       pixel_t *h_odata)
{
    int i, j, count = 0;
    #pragma omp parallel for num_threads(NUM_THREADS) private(i,j) reduction(+:count)
    for (i = 3; i < h - 3; i++) // height image
    {
        for (j = 3; j < w - 3; j++) // width image
        {
            if (fastCorner(h_idata + i * w + j, w, h, th_count, th_diff))
            {
                h_odata[i * w + j] = MAX_BRIGHTNESS;
                count++;
            }
        }
    }
    printf("detected %d features\n", count);
}

// Copy of FastDetectorHost but calls the parallel version of the functions
void fastDetectorOpenMP(const pixel_t *h_idata, const int w, const int h,
                        const int th_count, // min count to detect corners size
                        const int th_diff,  // threshold diff to count
                        const bool nonmaxflag,
                        pixel_t *h_odata)
{
    int i, j; // indexes in image

    // initialize h_odata to zero
    memset(h_odata, 0, h*w*sizeof(pixel_t));

    makeOffsets(offset, w);

    // corner detection
    fastDetectCornersOPENMP(h_idata, w, h, th_count, th_diff, h_odata);


    if (nonmaxflag)
    {
        pixel_t *aux = (pixel_t *)malloc(w * h * sizeof(pixel_t));

        memcpy(aux, h_odata, w * h * sizeof(pixel_t));

        //non max supression
        nonMaximumSupressionOPENMP(h_idata, aux, h_odata, w, h, th_count);

        free(aux);
    }

    //add faded original image as background
    #pragma omp parallel for num_threads(NUM_THREADS) private(i,j)
    for (i = 0; i < h; i++) // height image
    {
        for (j = 0; j < w; j++) // width image
        {
            if(h_odata[i * w + j]!=MAX_BRIGHTNESS) {
                h_odata[i * w + j] = h_idata[i * w + j] / FADEDIV; // to obtain a faded background image
            }
        }
    }
}


// print command line format
void usage(char *command)
{
    printf("Usage: %s [-h] [-i inputfile] [-o outputfile] [-r referenceFile] [-c th_count] [-t th_diff] [-m]\n", command);
}

// main
int main(int argc, char **argv)
{

    // default command line options
    char *fileIn = (char *)"house.pgm",
         *fileOut = (char *)"resultOpenMP.pgm",
         *referenceOut = (char *)"referenceOpenMP.pgm";
    unsigned int th_count = 9, th_diff = 50, nonmaxflag = 0;

    // parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:o:r:c:t:mh")) != -1)
    {
        switch (opt)
        {

        case 'i':
            if (strlen(optarg) == 0)
            {
                usage(argv[0]);
                exit(1);
            }

            fileIn = strdup(optarg);
            break;
        case 'o':
            if (strlen(optarg) == 0)
            {
                usage(argv[0]);
                exit(1);
            }
            fileOut = strdup(optarg);
            break;
        case 'r':
            if (strlen(optarg) == 0)
            {
                usage(argv[0]);
                exit(1);
            }
            referenceOut = strdup(optarg);
            break;
        case 'c':
            if (strlen(optarg) == 0 || sscanf(optarg, "%d", &th_count) != 1)
            {
                usage(argv[0]);
                exit(1);
            }
            break;
        case 't':
            if (strlen(optarg) == 0 || sscanf(optarg, "%d", &th_diff) != 1)
            {
                usage(argv[0]);
                exit(1);
            }
            break;
        case 'm':
            nonmaxflag = 1;
            break;
        case 'h':
            usage(argv[0]);
            exit(0);
            break;
        }
    }

    // allocate host memory
    pixel_t *h_idata = NULL;
    unsigned int h, w;

    // load pgm
    if (sdkLoadPGM<pixel_t>(fileIn, &h_idata, &w, &h) != true)
    {
        printf("Failed to load image file: %s\n", fileIn);
        exit(1);
    }

    // allocate mem for the result on host side
    pixel_t *h_odata = (pixel_t *)malloc(h * w * sizeof(pixel_t));
    pixel_t *reference = (pixel_t *)malloc(h * w * sizeof(pixel_t));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // detect corners
    fastDetectorHost(h_idata, w, h, th_count, th_diff, nonmaxflag, reference);
    
    gettimeofday(&end, NULL);

    struct timeval startMP, endMP;
    gettimeofday(&startMP, NULL);

    // detect corners with OpenMP
    fastDetectorOpenMP(h_idata, w, h, th_count, th_diff, nonmaxflag, h_odata);

    gettimeofday(&endMP, NULL);

    printf("\nTiming results:\n");
    printf("Host processing time: %f (ms)\n", (end.tv_sec - start.tv_sec) * 1000.0 + ((double)(end.tv_usec - start.tv_usec)) / 1000.0);
    printf("OpenMP processing time (using %i threads): %f (ms)\n", NUM_THREADS,(endMP.tv_sec - startMP.tv_sec) * 1000.0 + ((double)(endMP.tv_usec - startMP.tv_usec)) / 1000.0);

    double speedup = ((end.tv_sec - start.tv_sec) * 1000.0 + ((double)(end.tv_usec - start.tv_usec)) / 1000.0)/((endMP.tv_sec - startMP.tv_sec) * 1000.0 + ((double)(endMP.tv_usec - startMP.tv_usec)) / 1000.0);
    printf("Speedup: %f\n", speedup);

    // save output images
    if (sdkSavePGM<pixel_t>(referenceOut, reference, w, h) != true)
    {
        printf("Failed to save image file: %s\n", referenceOut);
        exit(1);
    }
    if (sdkSavePGM<pixel_t>(fileOut, h_odata, w, h) != true)
    {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
}
