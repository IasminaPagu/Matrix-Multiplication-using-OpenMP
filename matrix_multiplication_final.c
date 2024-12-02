#include<stdio.h>
#include<string.h>
#include<omp.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>

#define EPSILON 0.1
#define MAXRANGE 100
#define n 3000 //nr de linii si coloane
#define CHUNKSIZE 100
#define NTHREADS 12

long long int a[n][n];
long long int b[n][n];
long long int matrice_referinta[n][n];
long long int c[n][n];
long long int c2[n][n];


void initializare_matrice(){
    for(int i = 0;i<n;i++){
        for(int j = 0;j<n;j++){
            c[i][j] = 0;
        }
    }
}
//varianta seriala pentru i-j-k
void matrix_multiplication_serial_v1(){
    // initializare_matrice();
    for(int i = 0;i<n;i++){
        for(int j = 0;j<n;j++){
            
            matrice_referinta[i][j] = 0;
            for(int k = 0;k<n;k++){
                matrice_referinta[i][j] = matrice_referinta[i][j] + a[i][k] * b[k][j];
            }
        }
    }
}
//varianta seriala pentru i-k-j
void matrix_multiplication_serial_v2(){
    double aik;
    initializare_matrice();
    for(int i = 0;i<n;i++){
        for(int k = 0;k<n;k++){
            aik = a[i][k];
            for(int j = 0;j<n;j++){
                c[i][j] +=  aik * b[k][j];
            }
        }
    }
}

////varianta seriala pentru j-i-k
void matrix_multiplication_serial_v3(){
    initializare_matrice();
    for(int j = 0;j<n;j++){
        for(int i = 0;i<n;i++){
            for(int k = 0;k<n;k++){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
////varianta seriala pentru j-k-i
void matrix_multiplication_serial_v4(){
    double bjk;
    initializare_matrice();
    for(int j = 0;j<n;j++){
        for(int k = 0;k<n;k++){
            bjk = b[k][j];
            for(int i = 0;i<n;i++){
                c[i][j] += bjk * a[i][k];
            }
        }
    }
}
////varianta seriala pentru k-i-j
void matrix_multiplication_serial_v5(){
    double aik;
    initializare_matrice();

    for(int k = 0;k<n;k++){
        for(int i = 0;i<n;i++){
            aik = a[i][k];
            for(int j = 0;j<n;j++){
                c[i][j] += aik * b[k][j];
            }
        }
    }
}

////varianta seriala pentru k-j-i
void matrix_multiplication_serial_v6(){
    double bjk;
    initializare_matrice();

    for(int k = 0;k<n;k++){
        for(int j = 0;j<n;j++){
            bjk = b[k][j];
            for(int i = 0;i<n;i++){
                c[i][j] +=  bjk * a[i][k];
            }
        }
    }
}
//varianta paralele pentru i-j-k
void parallel_multiply_v1(int nthreads, int chunk){
    int i,j,k;
    #pragma omp parallel num_threads(nthreads), default(none), private(i, j), shared(c2, chunk)
    #pragma omp for schedule(static, chunk)
        for (i = 0; i < n; i++){
            for (j = 0; j < n; j++){
                    c2[i][j] = 0;
            }
        }
    #pragma omp parallel num_threads(nthreads), private(i,j,k),shared(a, b, c2, chunk)
    {
        #pragma omp for schedule(static, chunk)
            for(i = 0;i<n;i++){
                for(j = 0;j<n;j++){
                    c2[i][j] = 0;
                    for(k = 0;k<n;k++){
                        c2[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
    }
}
//varianta paralela pentru i-k-j
void parallel_multiply_v2(int nthreads, int chunk){
    int i,j,k;
    double aik;
    #pragma omp parallel num_threads(nthreads), default(none), private(i, j), shared(c2, chunk)
    #pragma omp for schedule(static, chunk)
        for (i = 0; i < n; i++){
            for (j = 0; j < n; j++){
                    c2[i][j] = 0;
            }
        }
                
    #pragma omp parallel num_threads(nthreads), private(i,j,k,aik),shared(a, b, c2, chunk)
    {
        #pragma omp for schedule(static,chunk)
            for(i = 0;i<n;i++){
                for(k = 0;k<n;k++){
                    aik = a[i][k];
                    for(j = 0;j<n;j++){
                        c2[i][j] +=  aik * b[k][j];
                    }
                }
            }
    }
}

//varianta paralela pentru j-i-k
void parallel_multiply_v3(int nthreads, int chunk){
    int i,j,k;
    #pragma omp parallel num_threads(nthreads), default(none), private(i, j), shared(c2, chunk)
    #pragma omp for schedule(static, chunk)
        for (i = 0; i < n; i++){
            for (j = 0; j < n; j++){
                    c2[i][j] = 0;
            }
        }
                
    #pragma omp parallel num_threads(nthreads), private(i,j,k),shared(a, b, c2, chunk)
    {
        #pragma omp for schedule(static,chunk)
            for(j = 0;j<n;j++){
                for(i = 0;i<n;i++){
                    for(k = 0;k<n;k++){
                        c2[i][j] +=  a[i][k] * b[k][j];
            }
        }
    }
    }
}
//varianta paralela pentru j-k-i
void parallel_multiply_v4(int nthreads, int chunk){
    int i,j,k;
    double bjk;
    #pragma omp parallel num_threads(nthreads), default(none), private(i, j), shared(c2, chunk)
    #pragma omp for schedule(static, chunk)
        for (i = 0; i < n; i++){
            for (j = 0; j < n; j++){
                    c2[i][j] = 0;
            }
        }
    #pragma omp parallel num_threads(nthreads), private(i,j,k,bjk),shared(a, b, c2, chunk)
    {
        #pragma omp for schedule(static,chunk)
            for(j = 0;j<n;j++){
                for(k = 0;k<n;k++){
                    bjk = b[k][j];
                    for(i = 0;i<n;i++){
                        c2[i][j] +=  bjk * a[i][k];
                    }
                }
            }
    }
}
//varianta paralela pentru k-i-j
void parallel_multiply_v5(int nthreads, int chunk){
    int i,j,k;
    double aik;
    #pragma omp parallel num_threads(nthreads), default(none), private(i, j), shared(c2, chunk)
    #pragma omp for schedule(static, chunk)
        for (i = 0; i < n; i++){
            for (j = 0; j < n; j++){
                    c2[i][j] = 0;
            }
        }
    #pragma omp parallel num_threads(nthreads), private(i,j,k,aik),shared(a, b, c2, chunk)
    {
        #pragma omp for schedule(static,chunk)
            for(k = 0;k<n;k++){
                for(i = 0;i<n;i++){
                    aik = a[i][k];
                    for(j = 0;j<n;j++){
                        c2[i][j] += aik * b[k][j];
                    }
                }
            }
    }
}

//varianta paralela pentru k-j-i
void parallel_multiply_v6(int nthreads, int chunk){
    int i,j,k;
    double bjk;
    #pragma omp parallel num_threads(nthreads), default(none), private(i, j), shared(c2, chunk)
    #pragma omp for schedule(static, chunk)
        for (i = 0; i < n; i++){
            for (j = 0; j < n; j++){
                    c2[i][j] = 0;
            }
        }
    #pragma omp parallel num_threads(nthreads), private(i,j,k,bjk),shared(a, b, c2, chunk)
    {
        #pragma omp for schedule(static,chunk)
            for(k = 0;k<n;k++){
                for(j = 0;j<n;j++){
                    bjk = b[k][j];
                    for(i = 0;i<n;i++){
                    c2[i][j] += bjk * a[i][k];
                    }
                }
            }
    }
}
void afisare_matrice(long long int c[n][n]){
    for(int i = 0;i<n;i++){
        for(int j = 0;j<n;j++){
            printf("%lld ",c[i][j]);
        }
        printf("\n");
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
int main(){

    double start, end, time_serial, time_parallel;
    
    printf("generate matrix a ...\n");
    Generate_matrix(a);

    printf("generate matrix b ...\n");
    Generate_matrix(b);

    // printf("matricea a este \n");
    // afisare_matrice(a);
    // printf("\nmatricea b este \n");
    // afisare_matrice(b);
    // printf("\n");
    
    printf("Start working serial V1 ... \n");
    start = omp_get_wtime();
    matrix_multiplication_serial_v1();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time V1 %lf seconds \n\n", time_serial);
    //aici o sa mi se contruiasca matricea mea de referinta
    //---------------------------------------------------------------

    printf("Start working parallel V1 with %d threads ... \n", NTHREADS);
    start = omp_get_wtime();
    parallel_multiply_v1(NTHREADS,CHUNKSIZE);
    end = omp_get_wtime();
    time_parallel = (end - start);

    printf("Parallel time V1 %lf seconds \n", time_parallel);
    printf("Speedup = %2.2lf\n\n", time_serial / time_parallel);
    if(Equal_matrixes(matrice_referinta,c2) == 0){
        printf("matricea obtinuta prin metoda parallel_v1 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c2);
    }
    //-------------------------------------------------------------------
    printf("Start working serial V2 ... \n");
    start = omp_get_wtime();
    matrix_multiplication_serial_v2();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time V2 %lf seconds \n\n", time_serial);

    if(Equal_matrixes(matrice_referinta,c) == 0){
        printf("matricea obtinuta prin metoda serial_v2 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c);
    }
    //------------------------------------------------------------------
    printf("Start working parallel V2 with %d threads ... \n", NTHREADS);
    start = omp_get_wtime();
    parallel_multiply_v2(NTHREADS,CHUNKSIZE);
    end = omp_get_wtime();
    time_parallel = (end - start);

    printf("Parallel time V2 %lf seconds \n", time_parallel);
    printf("Speedup = %2.2lf\n\n", time_serial / time_parallel);
    if(Equal_matrixes(matrice_referinta,c2) == 0){
        printf("matricea obtinuta prin metoda parallel_v2 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c2);
    }
    //-------------------------------------------------------------------
    printf("Start working serial V3 ... \n");
    start = omp_get_wtime();
    matrix_multiplication_serial_v3();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time V3 %lf seconds \n\n", time_serial);

    if(Equal_matrixes(matrice_referinta,c) == 0){
        printf("matricea obtinuta prin metoda serial_v3 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c);
    }
    //------------------------------------------------------------------
    printf("Start working parallel V3 with %d threads ... \n", NTHREADS);
    start = omp_get_wtime();
    parallel_multiply_v3(NTHREADS,CHUNKSIZE);
    end = omp_get_wtime();
    time_parallel = (end - start);

    printf("Parallel time V3 %lf seconds \n", time_parallel);
    printf("Speedup = %2.2lf\n\n", time_serial / time_parallel);
    if(Equal_matrixes(matrice_referinta,c2) == 0){
        printf("matricea obtinuta prin metoda parallel_v3 nu este corecta\n");
    //     printf("\nmatricea referinta este \n");
    //     afisare_matrice(matrice_referinta);

    //     printf("\n matricea mea este \n");
    //     afisare_matrice(c2);
    // }
    }
    //-----------------------------------------------------------------
    printf("Start working serial V4 ... \n");
    start = omp_get_wtime();
    matrix_multiplication_serial_v4();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time V4 %lf seconds \n\n", time_serial);

    if(Equal_matrixes(matrice_referinta,c) == 0){
        printf("matricea obtinuta prin metoda serial_v4 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c);
    }

    //------------------------------------------------------------------
    printf("Start working parallel V4 with %d threads ... \n", NTHREADS);
    start = omp_get_wtime();
    parallel_multiply_v4(NTHREADS,CHUNKSIZE);
    end = omp_get_wtime();
    time_parallel = (end - start);

    printf("Parallel time V4 %lf seconds \n", time_parallel);
    printf("Speedup = %2.2lf\n\n", time_serial / time_parallel);
    if(Equal_matrixes(matrice_referinta,c2) == 0){
        printf("matricea obtinuta prin metoda parallel_v4 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c2);
    }
    //-------------------------------------------------------------------
    printf("Start working serial V5 ... \n");
    start = omp_get_wtime();
    matrix_multiplication_serial_v5();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time V5 %lf seconds \n\n", time_serial);

    if(Equal_matrixes(matrice_referinta,c) == 0){
        printf("matricea obtinuta prin metoda serial_v5 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c);
    }
    //------------------------------------------------------------------
    printf("Start working parallel V5 with %d threads ... \n", NTHREADS);
    start = omp_get_wtime();
    parallel_multiply_v5(NTHREADS,CHUNKSIZE);
    end = omp_get_wtime();
    time_parallel = (end - start);

    printf("Parallel time V5 %lf seconds \n", time_parallel);
    printf("Speedup = %2.2lf\n\n", time_serial / time_parallel);
    if(Equal_matrixes(matrice_referinta,c2) == 0){
        printf("matricea obtinuta prin metoda parallel_v5 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c2);
    }
    //-------------------------------------------------------------------
    printf("Start working serial V6 ... \n");
    start = omp_get_wtime();
    matrix_multiplication_serial_v6();
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial time V6 %lf seconds \n\n", time_serial);

    if(Equal_matrixes(matrice_referinta,c) == 0){
        printf("matricea obtinuta prin metoda serial_v6 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c);
    }
    //------------------------------------------------------------------
    printf("Start working parallel V6 with %d threads ... \n", NTHREADS);
    start = omp_get_wtime();
    parallel_multiply_v6(NTHREADS,CHUNKSIZE);
    end = omp_get_wtime();
    time_parallel = (end - start);

    printf("Parallel time V6 %lf seconds \n", time_parallel);
    printf("Speedup = %2.2lf\n\n", time_serial / time_parallel);
    if(Equal_matrixes(matrice_referinta,c2) == 0){
        printf("matricea obtinuta prin metoda parallel_v6 nu este corecta\n");
        // printf("\nmatricea referinta este \n");
        // afisare_matrice(matrice_referinta);

        // printf("\n matricea mea este \n");
        // afisare_matrice(c2);
    }
    //---------------------------------------------------------------


    return 0;
}