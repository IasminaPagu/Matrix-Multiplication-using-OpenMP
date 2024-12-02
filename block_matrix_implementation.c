#include<stdio.h>
#include<string.h>
#include<omp.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>

#define EPSILON 0.1
#define MAXRANGE 100
#define n 1000 //nr de linii si coloane
int blocksize=100;

long long int a[n][n];
long long int b[n][n];
long long int matrice_referinta[n][n];
long long int c[n][n];

void initializare_matrice(long long int matrice[n][n]){
    for(int i = 0;i<n;i++){
        for(int j = 0;j<n;j++){
            matrice[i][j] = 0;
        }
    }
}
int val_min(int a, int b){
    if(a<b){
        return a;
    }else{
        return b;
    }
}
void Generate_matrix(long long int mat[n][n])
{
    int i, j;

    srand(time(NULL));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            mat[i][j] = (rand() % (2 * MAXRANGE)) - MAXRANGE;
}

int Equal_matrixes(long long int mat1[n][n],long long int mat2[n][n])
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
            if (fabs(mat1[i][j] - mat2[i][j] > EPSILON))
            {
                return 0;
            }
    }
    return 1;
}

void matrix_multiplication_serial_v1(){
    initializare_matrice(matrice_referinta);
    for(int i = 0;i<n;i++){
        for(int j = 0;j<n;j++){
            
            matrice_referinta[i][j] = 0;
            for(int k = 0;k<n;k++){
                matrice_referinta[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
void afisare_matrice(long long int matrice[n][n]){
    for(int i = 0;i<n;i++){
        for(int j = 0;j<n;j++){
            printf("%lld ",matrice[i][j]);
        }
        printf("\n");
    }
}
//Take into account also the case when  the size of the matrix is not
// evenly divisible to the block size. 
//am facut asta prin introducerea functiei val_min

void blocked_implementation_serial() {
    initializare_matrice(c);
    for (int kk = 0; kk < n; kk += blocksize) {
        for (int jj = 0; jj < n; jj += blocksize) {
            for (int i = 0; i < n; i++) {
                for (int j = jj; j < val_min(jj + blocksize, n); ++j) {  
                    for (int k = kk; k < val_min(kk + blocksize, n); ++k) { 
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    }
}
void blocked_implementation_parallel(){
    int kk,jj,i,j,k;       
    #pragma omp parallel default(none), private(i, j), shared(c, blocksize)
    #pragma omp for schedule(static, blocksize)
        for (i = 0; i < n; i++){
            for (j = 0; j < n; j++){
                    c[i][j] = 0;
            }
        }
    #pragma omp parallel private(kk,jj,i,j,k),shared(a, b, c, blocksize)
    {
        #pragma omp for schedule(static,blocksize)
            for (kk = 0; kk < n; kk += blocksize) {
                for (jj = 0; jj < n; jj += blocksize) {
                    for (i = 0; i < n; i++) {
                        for (j = jj; j < val_min(jj + blocksize, n); ++j) {  
                            for (k = kk; k < val_min(kk + blocksize, n); ++k) { 
                                c[i][j] += a[i][k] * b[k][j];
                            }
                        }
                    }
                }
            }
    }
}
int main(){

    double start, end, time_serial, time_parallel;
    
    printf("generate matrix a ...\n");
    Generate_matrix(a);

    printf("generate matrix b ...\n");
    Generate_matrix(b);


    printf("Start working serial V1, determinarea matricii de comparare... \n");
    start = omp_get_wtime();
    matrix_multiplication_serial_v1();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time V1 %lf seconds \n\n", time_serial);

    
    printf("Start working blocked serial implementation with blocksize = %d\n",blocksize);
    start = omp_get_wtime();
    blocked_implementation_serial();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Blocked serial time V1 %lf seconds \n\n", time_serial);

    if(Equal_matrixes(matrice_referinta,c) == 0){   
        printf("matricile nu sunt identice\n");
        printf("matricea referinta este \n");
        afisare_matrice(matrice_referinta);
        printf("matricea c este\n");
        afisare_matrice(c);
    }
    //---------------------------------------------------------------
    printf("Start working blocked parallel implementation with blocksize = %d\n",blocksize);
    start = omp_get_wtime();
    blocked_implementation_parallel();
    end = omp_get_wtime();
    time_parallel = end - start;
    printf("Blocked parallel time V1 %lf seconds \n\n", time_parallel);

    if(Equal_matrixes(matrice_referinta,c) == 0){   
        printf("matricile nu sunt identice\n");
        printf("matricea referinta este \n");
        //afisare_matrice(matrice_referinta);
        printf("matricea c este\n");
        //afisare_matrice(c);
    }

    return 0;
}