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
  Usage:  mpirun -np p ./program   (asks for n from the keyboard)
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
	int menum, nproc, tag, resto, j, rep, passi = 0, p_temp;
	int r, sendTo, recvBy, dist, *potenze;
	long long n, nloc, nlocgen, i;
	DATA_TYPE *vett_loc;
	double sommaloc, tmp, T_inizio, T_fine, T_max;
	(void)r; (void)sendTo; (void)recvBy; (void)dist; (void)tag; (void)rep;
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

	/* Read n (P0 only) and send it to everyone */
	if (menum == 0)
	{
		printf("Enter the number of elements to add: \n");
		fflush(stdout);
		if (scanf("%lld", &n) != 1 || n < nproc)
		{
			printf("Error: n must be an integer greater than or equal to the number of processes.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

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

	/* Measurement (just one, as in the original version) */
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

	}

	if (menum == 0)
	{
		printf("\nProcessors used (Strategy III): %d\n", nproc);
		printf("\nThe total sum is: %.6f\n", sommaloc);
		printf("\nLocal computation time (P0): %f\n", T_fine);
		printf("\nMaximum time (MPI_Reduce): %f\n", T_max);
	}

	free(vett_loc);
	free(potenze);
	MPI_Finalize();
	return 0;
}