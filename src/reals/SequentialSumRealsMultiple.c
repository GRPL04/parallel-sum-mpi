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
  Usage:  mpirun -np 1 ./program   (runs through n = 10^8 ... 3.2·10^9 by itself)
  Prints one row per n:  N  Time_1  Time_2  Time_3  Sum
-------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#ifndef DATA_TYPE
#define DATA_TYPE float               /* type used to store the data          */
#endif
#ifndef N_INITIAL
#define N_INITIAL        100000000LL  /* first value of n (10^8)             */
#endif
#ifndef NUM_N_VALUES
#define NUM_N_VALUES    6            /* 10^8 ... 3.2·10^9 (x2 each time)     */
#endif
#ifndef REPETITIONS
#define REPETITIONS     3            /* measurements for each n             */
#endif

/* Print a number with a decimal comma, ready to paste into Excel */
static void print_num(double t)
{
	char buf[64];
	snprintf(buf, sizeof buf, "%.6f", t);
	char *dot = strchr(buf, '.');
	if (dot) *dot = ',';
	printf("\t%s", buf);
}

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
	int k, rep;
	double times[REPETITIONS];

	MPI_Init(&argc, &argv);
	srand((unsigned int) time(0));

	printf("\n=== Sequential sum of reals ===\n");
	printf("N\tTime_1\tTime_2\tTime_3\tSum\n");
	fflush(stdout);

	n = N_INITIAL;
	for (k = 0; k < NUM_N_VALUES; k++, n *= 2)
	{

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

		for (rep = 0; rep < REPETITIONS; rep++)
		{

			somma = 0.0;
			T_inizio = MPI_Wtime();
			for (i = 0; i < n; i++)
				somma = somma + vett[i];       /* accumulated in double */
			T_fine = MPI_Wtime() - T_inizio;
			times[rep] = T_fine;
		}

		printf("%lld", n);
		for (rep = 0; rep < REPETITIONS; rep++) print_num(times[rep]);
		print_num(somma);
		printf("\n");
		fflush(stdout);

		free(vett);
	}

	MPI_Finalize();
	return 0;
}