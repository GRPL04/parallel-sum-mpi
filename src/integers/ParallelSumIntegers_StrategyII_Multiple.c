/* --------------------------------------------------------------------------
  PURPOSE
  Computation of the sum of n integers on a parallel computer of type
  MIMD with p processors (p a power of 2), Strategy II (binary tree:
  the total sum ends up in P0).
--------------------------------------------------------------------------
  VERSION USED FOR THE MEASUREMENT TABLES
  - It no longer asks for n from the keyboard: it automatically runs through
        n = 10^8, 2·10^8, 4·10^8, 8·10^8, 1.6·10^9, 3.2·10^9
  - For each n it generates and distributes the data ONCE and measures the sum
    REPETITIONS (=3) times on the same data.
  - p is set when launching the program:  mpirun -np p ./ParallelSum_StrategyII
  - Prints one row per n:  N  Time_1  Time_2  Time_3  Sum
    (tab-separated, ready to copy and paste into Excel).
  - Since 3.2·10^9 > INT_MAX, n and the indices are long long.
  - P0 generates the data of each process block by block and sends them at once,
    so it never stores the whole vector (saves memory for large n).
----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
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

static void fail(int menum, const char *msg)
{
	fprintf(stderr, "[Process %d] ERROR: %s\n", menum, msg);
	MPI_Abort(MPI_COMM_WORLD, 1);
}

int main (int argc, char **argv)
{
	/* variable declarations */
	int menum, nproc, tag, resto, dest;
	long long n, nloc, nlocgen, cnt, i;
	int *vett_loc, *buffer;
	long long sommaloc, tmp;
	int p, passi = 0, r, sendTo, recvBy, *potenze;
	int k, rep;
	double T_inizio, T_fine, T_max, times[REPETITIONS];

	MPI_Status info;

	/* Initialise the MPI environment */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &menum);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	/* Strategy II requires nproc to be a power of 2 */
	if ((nproc & (nproc - 1)) != 0)
	{
		if (menum == 0) fprintf(stderr, "The number of processes must be a power of 2.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* Compute passi = log_2(nproc) and the vector of powers of 2 */
	p = nproc;
	while (p != 1)
	{
		p = p >> 1;
		passi++;
	}
	potenze = (int*)calloc(passi + 1, sizeof(int));
	for (i = 0; i <= passi; i++)
	{
		potenze[i] = p << i;
	}

	if (menum == 0)
	{
		printf("\n=== Parallel sum - Strategy II | p = %d ===\n", nproc);
		printf("N\tTime_1\tTime_2\tTime_3\tSum\n");
		fflush(stdout);
		srand((unsigned int) time(0));
	}

	n = N_INITIAL;
	for (k = 0; k < NUM_N_VALUES; k++, n *= 2)
	{
		/* ---------- Load distribution ---------- */
		nlocgen = n / nproc;
		resto   = (int)(n % nproc);
		nloc    = (menum < resto) ? nlocgen + 1 : nlocgen;

		/* MPI_Send/MPI_Recv use an int count */
		if (nproc > 1 && nlocgen + 1 > INT_MAX)
			fail(menum, "local block too large for MPI_Send (use more processes).");

		vett_loc = (int*)malloc(nloc * sizeof(int));
		if (vett_loc == NULL)
			fail(menum, "not enough memory for vett_loc (reduce NUM_N_VALUES).");

		/* ---------- Generation and distribution (not timed) ---------- */
		if (menum == 0)
		{
			/* Addends of P0 */
			for (i = 0; i < nloc; i++)
			{
				vett_loc[i] = (int)rand() % 5 - 2;
			}

			/* Addends of the other processes: generated block by block and sent */
			if (nproc > 1)
			{
				buffer = (int*)malloc((nlocgen + 1) * sizeof(int));
				if (buffer == NULL)
					fail(menum, "not enough memory for the send buffer.");

				for (dest = 1; dest < nproc; dest++)
				{
					cnt = (dest < resto) ? nlocgen + 1 : nlocgen;
					for (i = 0; i < cnt; i++)
					{
						buffer[i] = (int)rand() % 5 - 2;
					}
					tag = dest; /* tag equal to the ID of the receiving process */
					MPI_Send(buffer, (int)cnt, MPI_INT, dest, tag, MPI_COMM_WORLD);
				}
				free(buffer);
			}
		}
		else
		{
			tag = menum;
			MPI_Recv(vett_loc, (int)nloc, MPI_INT, 0, tag, MPI_COMM_WORLD, &info);
		}

		/* ---------- REPETITIONS measured on the same data ---------- */
		for (rep = 0; rep < REPETITIONS; rep++)
		{
			sommaloc = 0;

			MPI_Barrier(MPI_COMM_WORLD);
			T_inizio = MPI_Wtime();

			/* Local partial sum */
			for (i = 0; i < nloc; i++)
			{
				sommaloc = sommaloc + vett_loc[i];
			}

			/* --- STRATEGY 2 (BINARY TREE TOWARDS P0) --- */
			for (i = 0; i < passi; i++)
			{
				r = menum % potenze[i + 1];

				if (r == potenze[i])        /* menum sends */
				{
					sendTo = menum - potenze[i];
					tag = sendTo;
					MPI_Send(&sommaloc, 1, MPI_LONG_LONG, sendTo, tag, MPI_COMM_WORLD);
				}
				else if (r == 0)            /* menum receives */
				{
					recvBy = menum + potenze[i];
					tag = menum;
					MPI_Recv(&tmp, 1, MPI_LONG_LONG, recvBy, tag, MPI_COMM_WORLD, &info);
					sommaloc = sommaloc + tmp;
				}
			}

			MPI_Barrier(MPI_COMM_WORLD);
			T_fine = MPI_Wtime() - T_inizio;

			/* Execution time = maximum over the processes */
			MPI_Reduce(&T_fine, &T_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			if (menum == 0) times[rep] = T_max;
		}

		/* ---------- Table row ---------- */
		if (menum == 0)
		{
			printf("%lld", n);
			for (rep = 0; rep < REPETITIONS; rep++) print_time(times[rep]);
			printf("\t%lld\n", sommaloc);
			fflush(stdout);
		}

		free(vett_loc);
	}

	free(potenze);

	/* Shut down the MPI environment */
	MPI_Finalize();
	return 0;
}
