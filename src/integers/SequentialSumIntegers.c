/* --------------------------------------------------------------------------
  PURPOSE
  Sequential version of the sum of n integers adapted with
  basic MPI calls to measure execution times as in the parallel case.
-------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main (int argc, char **argv)
{
	/* Variable declarations */
    int menum, nproc;
	int n, i, somma = 0;
	int *vett;

	double T_inizio, T_fine;

	MPI_Status info;

	/* Initialise the MPI environment */
	MPI_Init(&argc, &argv);

	/* Read the input data */
    printf("Enter the number of elements to add: \n");
	fflush(stdout);
	scanf("%d", &n);

       	vett = (int*)calloc(n, sizeof(int));

    srand((unsigned int) time(0));

    for(i = 0; i < n; i++)
    {
		/* Fill the vector with random numbers */
		*(vett + i) = (int)rand() % 5 - 2;
	}

   	/* Print the vector to be added if it has fewer than 100 elements */
	if (n < 100)
    {
		for (i = 0; i < n; i++)
        {
				printf("\n\nVector element %d = %d ", i, *(vett + i));
		}
    }

    T_inizio = MPI_Wtime(); // Start the timer

    for(i = 0; i < n; i++)
    {
        somma = somma + *(vett + i);
    }

    T_fine = MPI_Wtime() - T_inizio; // Elapsed time

    /* Print the final results */
	printf("\nThe sum is: %d\n", somma);
	printf("\nComputation time: %lf\n", T_fine);

	/* Shut down the MPI environment */
	MPI_Finalize();
}