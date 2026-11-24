/* --------------------------------------------------------------------------
  PURPOSE
  Sequential sum of n REAL numbers (MPI is used only for MPI_Wtime).
--------------------------------------------------------------------------
  CHANGES NEEDED TO RUN UP TO n = 3.2·10^9
  1. n and the indices are long long: 3.2·10^9 does not fit in an int (max. 2,147,483,647).
  2. The data are stored as DATA_TYPE (float by default, 4 bytes) and added up
     in double. With double, 3.2·10^9 values would take 25.6 GB; with float they take
     12.8 GB, the same as the integers. If the machine has enough memory,
     just change DATA_TYPE to double.
  3. The program allocates and generates the n values directly.
  4. Before measuring, an untimed warm-up sum is run so that
     all the data are already in memory and the first time is not inflated.
  Usage:  mpirun -np 1 ./program   (asks for n from the keyboard)
-------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#ifndef DATA_TYPE
#define DATA_TYPE float               /* type used to store the data          */
#endif

/* Random real number in [-2, 2] */
static DATA_TYPE random_value(void)
{
	return (DATA_TYPE)(((double)rand() / RAND_MAX) * 4.0 - 2.0);
}

static void fail(int menum, const char *msg)
{
	fprintf(stderr, "[Process %d] ERROR: %s\n", menum, msg);
	MPI_Abort(MPI_COMM_WORLD, 1);
}

int main (int argc, char **argv)
{
	long long n, i;
	DATA_TYPE *vett;
	double somma, T_inizio, T_fine;

	MPI_Init(&argc, &argv);
	srand((unsigned int) time(0));

	printf("Enter the number of elements to add: \n");
	fflush(stdout);
	if (scanf("%lld", &n) != 1 || n < 1)
	{
		printf("Error: n must be a positive integer.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	vett = (DATA_TYPE*)malloc(n * sizeof(DATA_TYPE));
	if (vett == NULL)
		fail(0, "not enough memory for the data.");

	/* Data generation (not timed) */
	for (i = 0; i < n; i++)
		vett[i] = random_value();

	/* Untimed warm-up sum */
	somma = 0.0;
	for (i = 0; i < n; i++)
		somma = somma + vett[i];

	{

		somma = 0.0;
		T_inizio = MPI_Wtime();
		for (i = 0; i < n; i++)
			somma = somma + vett[i];       /* accumulated in double */
		T_fine = MPI_Wtime() - T_inizio;
	}

	printf("\nThe sum is: %.6f\n", somma);
	printf("\nComputation time: %f\n", T_fine);

	free(vett);
	MPI_Finalize();
	return 0;
}