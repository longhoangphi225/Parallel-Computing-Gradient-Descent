#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
int n, p;

void read_data(double* &X, double* &X_T, double* &y, double* &theta, double* &gradient)
{
    FILE *f;
    f = fopen("5000_10.txt","r");
    fscanf(f,"%d %d", &n, &p);
    printf("Tinh toan voi du lieu n=%d va p=%d\n", n, p);
    X = new double[n * (p+1)];
    X_T =  new double[(p+1) * n];
    y = new double[n * 1];
    theta = new double[(p+1) * 1];
    gradient = new double[(p+1) * 1];
    for (int i=0; i<n; i++)
    {
        X[i*(p+1)] = 1.0;
        for (int j=1; j<p+2; j++)
        {
            if (j<p+1) fscanf(f,"%lf",&X[i*(p+1)+j]);
                else fscanf(f,"%lf \n",&y[i]);
        }
    }

    for (int i=0; i<p+1; i++)
        for (int j=0; j<n; j++)
            X_T[i*n+j] = X[j*(p+1)+i];

    fclose(f);
    f = fopen("theta_10.txt","r");
    for (int i=0; i<p+1; i++)
    {
        if (i<p) fscanf(f,"%lf\n",&theta[i]);
            else fscanf(f,"%lf",&theta[i]);
    }
    fclose(f);
}

int matmul_parallel(int n, int p, int m, double* A, double* B, double* C)
{
   int i,j,k;
# pragma omp parallel shared(A,B,C) private(i,j,k)
   {
# pragma omp for  schedule(static)
    for (i=0; i<n; i++)
            for (j=0; j<m; j++)
            {
                C[i*m+j] = 0;
                for (k=0; k<p; k++)
                    C[i*m+j] += A[i*p+k]*B[k*m+j];
            }
   }
   return 0;
}

int matmul_serial(int n, int p, int m, double* A, double* B, double* C)
{
   int i,j,k;
   {
    for (i=0; i<n; i++)
            for (j=0; j<m; j++)
            {
                C[i*m+j] = 0;
                for (k=0; k<p; k++)
                    C[i*m+j] += A[i*p+k]*B[k*m+j];
            }
   }
   return 0;
}

double norm_parallel(int p, double* &gradient)
{
    int i;
    double norm=0;


    # pragma omp parallel for reduction(+:norm)
    for (i=0; i<p+1; i++)
    {
        norm = gradient[i]*gradient[i];
    }

    return sqrt(norm);

}

double norm_serial(int p, double* &gradient)
{
    int i;
    double norm=0;

    for (i=0; i<p+1; i++)
    {
        norm += gradient[i]*gradient[i];
    }

    return sqrt(norm);

}

// example comment

void gradient_descent_parallel(double* &X, double* &X_T, double* &y, double* &theta, double* &gradient)
{
    double* X_theta;
    int i;
    double learning_rate=0.1, eps=0.0000000001;
    X_theta = new double[n*1];
    do
    {

        matmul_parallel(n, p+1, 1, X, theta, X_theta);
        # pragma omp parallel shared (X_theta, n) private (i)
        # pragma omp for
        for (i=0; i<n; i++)
            X_theta[i] -= y[i];

        matmul_parallel(p+1, n, 1, X_T, X_theta, gradient);

        # pragma omp parallel shared (gradient, p) private (i)
        # pragma omp for
        for (i=0; i<p+1; i++)
            gradient[i] = gradient[i]*2/n;

        # pragma omp parallel shared (gradient, p) private (i)
        # pragma omp for
        for (int i = 0; i<p+1; i++)
            theta[i] = theta[i] - learning_rate*gradient[i];
    }
    while (norm_parallel(p, gradient)>eps);
    delete[] X_theta;
}

void gradient_descent_serial(double* &X, double* &X_T, double* &y, double* &theta, double* &gradient)
{
    double* X_theta;
    int i;
    double learning_rate=0.1, eps=0.0000000001;
    X_theta = new double[n*1];
    do
    {
        matmul_serial(n, p+1, 1, X, theta, X_theta);
        for (i=0; i<n; i++)
            X_theta[i] -= y[i];

        matmul_serial(p+1, n, 1, X_T, X_theta, gradient);

        for (i=0; i<p+1; i++)
            gradient[i] = gradient[i]*2/n;

        for (int i = 0; i<p+1; i++)
            theta[i] = theta[i] - learning_rate*gradient[i];
    }
    while (norm_parallel(p, gradient)>eps);
    delete[] X_theta;
}

int main()
{
    double* X;
    double* X_T;
    double* y;
    double* theta;
    double* gradient;
    int num_processor;
    read_data(X, X_T, y, theta, gradient);

    double wtime1, wtime2;
    wtime1= omp_get_wtime ( );
    gradient_descent_serial(X, X_T, y, theta, gradient);
    wtime1 = omp_get_wtime ( ) - wtime1;
    printf ( "Thoi gian tinh toan tuan tu = %g\n", wtime1 );
    printf("Ket qua tinh toan tuan tu: \n");
    for (int i=0; i<p+1; i++)
    {
        printf("\t theta_%d = %lf \n",i, theta[i]);
    }
    wtime2 = omp_get_wtime ( );
    num_processor = omp_get_num_procs ( );
    gradient_descent_parallel(X, X_T, y, theta, gradient);
    wtime2 = omp_get_wtime ( ) - wtime2;
    printf ( "Thoi gian tinh toan song song = %g\n", wtime2);
    printf("Ket qua tinh toan song song: \n");
    for (int i=0; i<p+1; i++)
    {
        printf("\t theta_%d = %lf \n",i, theta[i]);
    }
    printf ( "\nHieu suat la: %lf\n", (100*wtime1/wtime2)/num_processor);
    delete[] X;
    delete[] X_T;
	delete[] y;
	delete[] theta;
	delete[] gradient;
}
