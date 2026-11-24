/* --------------------------------------------------------------------------
  PURPOSE
  Sequential version of the sum of n integers (MPI is used only for
  MPI_Wtime, as in the parallel versions).
--------------------------------------------------------------------------
  VERSION USED FOR THE MEASUREMENT TABLES
  - It no longer asks for n from the keyboard: it automatically runs through
        n = 10^8, 2·10^8, 4·10^8, 8·10^8, 1.6·10^9, 3.2·10^9
  - For each n it generates the data ONCE and measures the sum
    REPETITIONS (=3) times on the same data.
  - Run with a single process:  mpirun -np 1 ./SequentialSum
  - Prints one row per n:  N  Time_1  Time_2  Time_3  Sum
    (tab-separated, ready to copy and paste into Excel).
  - Since 3.2·10^9 > INT_MAX, n and the indices are long long.
-------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

/* ------------------- Test parameters ------------------- */
#ifndef N_INITIAL
#define N_INITIAL        100000000LL  /* first value of n (10^8)        */
#endif
#ifndef NUM_N_VALUES
#define NUM_N_VALUES    6            /* 10^8 ... 3.2·10^9 (x2 each time)  */
#endif
#ifndef REPETITIONS
#define REPETITIONS     3            /* runs for each n                */
#endif
#ifndef DECIMAL_COMMA
#define DECIMAL_COMMA 1            /* 1: 0,123456 (decimal comma)    */
#endif                                /* 0: 0.123456                     */

static void print_time(double t)
{
	char buf[64];
	snprintf(buf, sizeof buf, "%.6f", t);
	if (DECIMAL_COMMA)
	{
		char *dot = strchr(buf, '.');
		if (dot) *dot = ',';
	}
	printf("\t%s", buf);
}

int main (int argc, char **argv)
{
	/* Variable declarations */
	long long n, i, somma = 0;
	int *vett;
	int k, rep;
	double T_inizio, T_fine, times[REPETITIONS];

	/* Initialise the MPI environment */
	MPI_Init(&argc, &argv);

	printf("\n=== Sequential sum ===\n");
	printf("N\tTime_1\tTime_2\tTime_3\tSum\n");
	fflush(stdout);

	srand((unsigned int) time(0));

	n = N_INITIAL;
	for (k = 0; k < NUM_N_VALUES; k++, n *= 2)
	{
		vett = (int*)malloc(n * sizeof(int));
		if (vett == NULL)
		{
			fprintf(stderr, "ERROR: not enough memory for n = %lld (reduce NUM_N_VALUES).\n", n);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		/* Data generation (not timed) */
		for (i = 0; i < n; i++)
		{
			vett[i] = (int)rand() % 5 - 2;
		}

		/* REPETITIONS measured on the same data */
		for (rep = 0; rep < REPETITIONS; rep++)
		{
			somma = 0;

			T_inizio = MPI_Wtime();
			for (i = 0; i < n; i++)
			{
				somma = somma + vett[i];
			}
			T_fine = MPI_Wtime() - T_inizio;

			times[rep] = T_fine;
		}

		/* Table row */
		printf("%lld", n);
		for (rep = 0; rep < REPETITIONS; rep++) print_time(times[rep]);
		printf("\t%lld\n", somma);
		fflush(stdout);

		free(vett);
	}

	/* Shut down the MPI environment */
	MPI_Finalize();
	return 0;
}