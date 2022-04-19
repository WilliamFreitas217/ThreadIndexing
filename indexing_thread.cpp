#include <iostream>
#include <cstdio>

#include <vector>
#include <omp.h>
#include <sys/time.h>
#include <pthread.h>

using namespace std;

const int n = 4;
int mul_thread[n][n];
int mul_usual[n][n];
int mul_omp[n][n];
int matrix_1 [n][n];
int matrix_2 [n][n];

int row_i = 0;

void* multiplicate(void* data){
    int i = row_i++;

    for(int j=0;j<n;j++){
        mul_thread[i][j]=0;
        for(int k=0;k<n;k++){
            mul_thread[i][j]+=matrix_1[i][k]*matrix_2[k][j];
        }
    }
    return nullptr;
}

void usual_multiplicate(int mat1[][n], int mat2[][n]) {

    double st=omp_get_wtime();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mul_usual[i][j] = 0;
            for (int k = 0; k < n; k++) {
                mul_usual[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    double en=omp_get_wtime();
    printf("Usual multiplication Time: %lf\n",en-st);
}

void matrix_mult_parallel2(int mat1[][n], int mat2[][n])
{
    //Dynamic Scheduler
    int i,j,k;
    double st=omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,50) collapse(2) private(i,j,k) shared(matrix_1,matrix_2,mul_omp)
    for(i=0;i<n;i++){
        for( j=0;j<n;j++){
            for(k=0;k<n;k++){
                mul_omp[i][j]+=matrix_1[i][k]*matrix_2[k][j];
            }
        }
    }
    double en=omp_get_wtime();
    printf("OMP Time: %lf\n",en-st);
}

template<typename T>
void print(T mat) {
    for(int i=0 ; i<=n-1 ; i++) {
        for(int j=0 ; j<=n-1 ; j++)
            cout<< *(*(mat+i)+j)<<" ";
        cout<<endl;
    }
    cout<<endl;
}

int main() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_1[i][j] = random() % 10;
            matrix_2[i][j] = random() % 10;
        }
    }
    cout << "Matrix 1 \n" << endl;
    print(matrix_1);
    cout << "Matrix 2 \n" << endl;
    print(matrix_2);

    cout << "___________________________________\n" << endl;
    cout << "Usual Multiplication\n" << endl;

    usual_multiplicate(matrix_1, matrix_2);
    print(mul_usual);

    cout << "___________________________________\n" << endl;
    cout << "Thread Multiplication\n" << endl;


    double st=omp_get_wtime();

    pthread_t threads[n];

    for(int i=0;i<n;i++){
        int *p = nullptr;
        pthread_create(&threads[i], NULL, multiplicate, (void *) p);
    }

    for(int i=0;i<n;i++){
        pthread_join(threads[i], NULL);
    }

    double en=omp_get_wtime();
    print(mul_thread);
    printf("Thread time: %lf\n",en-st);

    cout << "___________________________________\n" << endl;
    cout << "OMP Multiplication\n" << endl;

    matrix_mult_parallel2(matrix_1, matrix_2);

    print(mul_omp);

}
