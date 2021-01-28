/* Host code that implements a  separable convolution filter of a 
 * 2D signal with a gaussian kernel.
 * 
 * Author: Naga Kandasamy
 * Date modified: May 26, 2020
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

extern "C" void compute_gold(float *, float *, int, int, int);
extern "C" float *create_kernel(float, int);
void print_kernel(float *, int);
void print_matrix(float *, int, int);
void compute_on_device(float *, float *, float *, int, int, int);
void check_for_error(const char *);

/* Width of convolution kernel */
#define HALF_WIDTH 8
#define COEFF 10

__constant__ float kernel_c[2 * HALF_WIDTH + 1]; /* Allocation for the kernel in GPU constant memory */


/* Uncomment line below to spit out debug information */
// #define DEBUG

#define THREAD_BLOCK 256

/* Include device code */
#include "separable_convolution_kernel.cu"

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s num-rows num-columns\n", argv[0]);
        printf("num-rows: height of the matrix\n");
        printf("num-columns: width of the matrix\n");
        exit(EXIT_FAILURE);
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);

    /* Create input matrix */
    int num_elements = num_rows * num_cols;
    printf("Creating input matrix of %d x %d\n", num_rows, num_cols);
    float *matrix_a = (float *)malloc(sizeof(float) * num_elements);
    float *matrix_c = (float *)malloc(sizeof(float) * num_elements);
	
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++) {
        matrix_a[i] = rand()/(float)RAND_MAX;			 
        matrix_c[i] = matrix_a[i]; /* Copy contents of matrix_a into matrix_c */
    }
	 
	/* Create Gaussian kernel */	  
    float *gaussian_kernel = create_kernel((float)COEFF, HALF_WIDTH);	
#ifdef DEBUG
    print_kernel(gaussian_kernel, HALF_WIDTH); 
#endif  

    /* Convolve matrix along rows and columns. 
       The result is stored in matrix_a, thereby overwriting the 
       original contents of matrix_a.		
     */
    printf("\nConvolving the matrix on the CPU\n");	  
    struct timeval start, stop;	
    gettimeofday(&start, NULL);

    compute_gold(matrix_a, gaussian_kernel, num_cols,\
                  num_rows, HALF_WIDTH);
    
    gettimeofday(&stop, NULL);
    printf("Execution time for the CPU= %f \n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
    
#ifdef DEBUG	 
    print_matrix(matrix_a, num_cols, num_rows);
#endif
  
    float *gpu_result = (float *)malloc(sizeof(float) * num_elements);
    
    /* FIXME: Edit this function to complete the functionality on the GPU.
       The input matrix is matrix_c and the result must be stored in 
       gpu_result.
     */
    printf("\nConvolving matrix on the GPU\n");
    compute_on_device(gpu_result, matrix_c, gaussian_kernel, num_cols,\
                       num_rows, HALF_WIDTH);
       
    printf("\nComparing CPU and GPU results\n");
    float sum_delta = 0, sum_ref = 0;
    for (i = 0; i < num_elements; i++) {
        sum_delta += fabsf(matrix_a[i] - gpu_result[i]);
        sum_ref   += fabsf(matrix_a[i]);
    }
        
    float L1norm = sum_delta / sum_ref;
    float eps = 1e-6;
    printf("L1 norm: %E\n", L1norm);
    printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

    free(matrix_a);
    free(matrix_c);
    free(gpu_result);
    free(gaussian_kernel);

    exit(EXIT_SUCCESS);
}
/* FIXME: Edit this function to compute the convolution on the device.*/
void compute_on_device(float *gpu_result, float *matrix_c, float *gaussian_kernel,\
                            int num_cols, int num_rows, int half_width)
{
    float *matrix_d = NULL;
    cudaMalloc((void**)&matrix_d, num_cols * num_rows * sizeof(float));
    cudaMemcpy(matrix_d, matrix_c, num_cols * num_rows * sizeof(float), cudaMemcpyHostToDevice);
    
    float *gaussian_kernel_d = NULL;
    cudaMalloc((void**)&gaussian_kernel_d, (2 * half_width + 1) * sizeof(float));
    cudaMemcpy(gaussian_kernel_d, gaussian_kernel, (2 * half_width + 1) * sizeof(float), cudaMemcpyHostToDevice);
    

    float *matrix_temp_d = NULL;
    cudaMalloc((void**)&matrix_temp_d, num_cols * num_rows * sizeof(float));
    
    float *gpu_result_d = NULL;
    cudaMalloc((void**)&gpu_result_d, num_cols * num_rows * sizeof(float));

    
    dim3 thread_block(THREAD_BLOCK, 1, 1);
    int num_block = (num_rows + THREAD_BLOCK - 1) / THREAD_BLOCK;
    dim3 grid(num_block, 1); 

    struct timeval start, stop;	
    gettimeofday(&start, NULL);
    
    printf("Using global memory for convolution\n");    
	convolve_rows_kernel_naive<<<grid, thread_block>>>(matrix_temp_d,  matrix_d, gaussian_kernel_d,\
                                                         num_cols,  num_rows,  half_width); 
    cudaDeviceSynchronize();
	check_for_error("KERNEL FAILURE ROW");
    
	convolve_columns_kernel_naive<<<grid, thread_block>>>(gpu_result_d, matrix_temp_d,  gaussian_kernel_d,\
                                                             num_cols,  num_rows,  half_width); 
    cudaDeviceSynchronize();
    gettimeofday(&stop, NULL);
	printf("Execution time for the GPU= %f \n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
    check_for_error("KERNEL FAILURE COLUMN");
    
    cudaMemcpy(gpu_result, gpu_result_d, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(matrix_d);
    cudaFree(matrix_temp_d);
    cudaFree(gpu_result_d);
    cudaFree(gaussian_kernel_d);
}

/* Check for errors reported by the CUDA run time */
void check_for_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    return;
} 

/* Print convolution kernel */
void print_kernel(float *kernel, int half_width)
{
    int i, j = 0;
    for (i = -half_width; i <= half_width; i++) {
        printf("%0.2f ", kernel[j]);
        j++;
    }

    printf("\n");
    return;
}

/* Print matrix */
void print_matrix(float *matrix, int num_cols, int num_rows)
{
    int i,  j;
    float element;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++){
            element = matrix[i * num_cols + j];
            printf("%0.2f ", element);
        }
        printf("\n");
    }

    return;
}

