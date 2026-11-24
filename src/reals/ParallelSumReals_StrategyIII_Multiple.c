/* --------------------------------------------------------------------------
  PURPOSE
  Parallel sum of n REAL numbers with Strategy III (butterfly).
  The number of processes must be a power of 2.
--------------------------------------------------------------------------
  CHANGES NEEDED TO RUN UP TO n = 3.2·10^9
  1. n and the indices are long long: 3.2·10^9 does not fit in an int (max. 2,147,483,647).
  2. The data are stored as DATA_TYPE (float by default, 4 bytes) and added up
     in double. With double, 3.2·10^9 values would take 25.6 GB; with float they take
     12.8 GB, the same as the integers. If the machine has enough memory,
     just change DATA_TYPE to double.
  3. Each process generates its own n/p values (with a different seed)
     instead of having P0 generate and send them: no extra buffer is needed
     in P0 and the total memory is exactly n·sizeof(DATA_TYPE). The distribution was not
     timed, so the measurement does not change.
  4. Before measuring, an untimed warm-up sum is run so that
     all the data are already in memory and the first time is not inflated.
  Usage:  mpirun -np p ./program   (runs through n = 10^8 ... 3.2·10^9 by itself)
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
	int menum, nproc, tag, resto, j, rep, passi = 0, p_temp;
	int r, sendTo, recvBy, dist, *potenze;
	long long n, nloc, nlocgen, i;
	DATA_TYPE *vett_loc;
	double sommaloc, tmp, T_inizio, T_fine, T_max;
	(void)r; (void)sendTo; (void)recvBy; (void)dist; (void)tag; (void)rep;
	int k;
	double times[REPETITIONS];
	MPI_Status info;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &menum);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	if ((nproc & (nproc - 1)) != 0)
	{
		if (menum == 0) fprintf(stderr, "The number of processes must be a power of 2.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* passi = log2(nproc) and powers of 2 (used by strategies II and III) */
	p_temp = nproc;
	while (p_temp > 1) { p_temp = p_temp >> 1; passi++; }
	potenze = (int*)calloc(passi + 1, sizeof(int));
	for (j = 0; j <= passi; j++) potenze[j] = 1 << j;

	/* Different seed in each process */
	srand((unsigned int) time(0) + 7919u * (unsigned int) menum);

	if (menum == 0)
	{
		printf("\n=== Parallel sum of reals - Strategy III | p = %d ===\n", nproc);
		printf("N\tTime_1\tTime_2\tTime_3\tSum\n");
		fflush(stdout);
	}

	n = N_INITIAL;
	for (k = 0; k < NUM_N_VALUES; k++, n *= 2)
	{

		/* Load distribution: the first "resto" processes get one extra value */
		nlocgen = n / nproc;
		resto   = (int)(n % nproc);
		nloc    = (menum < resto) ? nlocgen + 1 : nlocgen;

		vett_loc = (DATA_TYPE*)malloc(nloc * sizeof(DATA_TYPE));
		if (vett_loc == NULL)
			fail(menum, "not enough memory for the data.");

		/* Each process generates its own data (not timed) */
		for (i = 0; i < nloc; i++)
			vett_loc[i] = random_value();

		/* Untimed warm-up sum: leaves the data loaded in memory */
		sommaloc = 0.0;
		for (i = 0; i < nloc; i++)
			sommaloc = sommaloc + vett_loc[i];

		/* REPETITIONS measured on the same data */
		for (rep = 0; rep < REPETITIONS; rep++)
		{
			sommaloc = 0.0;
			MPI_Barrier(MPI_COMM_WORLD);
			T_inizio = MPI_Wtime();

			/* Local partial sum (accumulated in double) */
			for (i = 0; i < nloc; i++)
				sommaloc = sommaloc + vett_loc[i];

			/* --- STRATEGY 3 (BUTTERFLY / ALL-TO-ALL EXCHANGE) --- */
			for (j = 0; j < passi; j++)
			{
				dist = 1 << j;                  /* 2^j     */
				r = menum % (1 << (j + 1));     /* 2^(j+1) */
				if (r < dist)
				{
					sendTo = menum + dist;  recvBy = menum + dist;  tag = menum;
				}
				else
				{
					sendTo = menum - dist;  recvBy = menum - dist;  tag = sendTo;
				}
				MPI_Send(&sommaloc, 1, MPI_DOUBLE, sendTo, tag, MPI_COMM_WORLD);
				MPI_Recv(&tmp, 1, MPI_DOUBLE, recvBy, tag, MPI_COMM_WORLD, &info);
				sommaloc = sommaloc + tmp;
			}

			MPI_Barrier(MPI_COMM_WORLD);
			T_fine = MPI_Wtime() - T_inizio;

			/* Execution time = maximum over the processes */
			MPI_Reduce(&T_fine, &T_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

			if (menum == 0) times[rep] = T_max;
		}

		/* Table row */
		if (menum == 0)
		{
			printf("%lld", n);
			for (rep = 0; rep < REPETITIONS; rep++) print_num(times[rep]);
			print_num(sommaloc);
			printf("\n");
			fflush(stdout);
		}

		free(vett_loc);
	}

	free(potenze);
	MPI_Finalize();
	return 0;
}