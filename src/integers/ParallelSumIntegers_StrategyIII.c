/* --------------------------------------------------------------------------
  PURPOSE
  Computation of the sum of n integers in parallel using
  Strategy III (butterfly / symmetric exchange on a binary tree).
-------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main (int argc, char **argv)
{
    int menum, nproc, tag;
	int n, nloc, i, resto, nlocgen;
	int ind, k, dist, r, sendTo, recvBy, tmp;
	int *vett, *vett_loc;
	int sommaloc = 0;
	double T_inizio, T_fine, T_max;

	MPI_Status info;

	/* Initialise the MPI environment */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &menum);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	/* Read the input (process 0 only) */
	if (menum == 0)
	{
		printf("Enter the number of elements to add: \n");
		fflush(stdout);
		scanf("%d", &n);
		vett = (int*)calloc(n, sizeof(int));
	}

	/* Broadcast n to all processes */
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Workload distribution */
	nlocgen = n / nproc;
    resto = n % nproc;

	if (menum < resto)
	{
		nloc = nlocgen + 1;
	}
	else
	{
		nloc = nlocgen;
	}

	vett_loc = (int*)calloc(nloc, sizeof(int));

	if (menum == 0)
	{
        srand((unsigned int) time(0));
        for(i = 0; i < n; i++)
		{
			*(vett + i) = (int)rand() % 5 - 2;
		}

		if (n < 100)
		{
			for (i = 0; i < n; i++)
			{
				printf("\n\nVector element %d = %d ", i, *(vett + i));
			}
        }

        for(i = 0; i < nloc; i++)
		{
			*(vett_loc + i) = *(vett + i);
		}

		ind = nloc;

		for(i = 1; i < nproc; i++)
		{
			tag = i;
            if (i < resto)
			{
				MPI_Send(vett + ind, nloc, MPI_INT, i, tag, MPI_COMM_WORLD);
				ind = ind + nloc;
			}
			else
			{
				MPI_Send(vett + ind, nlocgen, MPI_INT, i, tag, MPI_COMM_WORLD);
				ind = ind + nlocgen;
			}
		}
	}
    else
    {
		tag = menum;
		MPI_Recv(vett_loc, nloc, MPI_INT, 0, tag, MPI_COMM_WORLD, &info);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	T_inizio = MPI_Wtime();

	/* Each processor computes its local partial sum */
	for(i = 0; i < nloc; i++)
	{
		sommaloc = sommaloc + *(vett_loc + i);
	}

	/* --- STRATEGY 3 (BUTTERFLY / ALL-TO-ALL EXCHANGE) --- */
    int passi = 0;
    int p_temp = nproc;
    while(p_temp > 1) {
        p_temp = p_temp >> 1;
        passi++;
    }

    for(k = 0; k < passi; k++)
    {
        dist = 1 << k;            // Dist = 2^k
        int mod_val = 1 << (k+1); // Modulus equal to 2^(k+1)
        r = menum % mod_val;

        // Structure based on the logical blocks of the scheme (k=0, k=1, k=2...)
        if (r < dist)
        {
            sendTo = menum + dist;
            recvBy = menum + dist;
            tag = menum;

            MPI_Send(&sommaloc, 1, MPI_INT, sendTo, tag, MPI_COMM_WORLD);
            MPI_Recv(&tmp, 1, MPI_INT, recvBy, tag, MPI_COMM_WORLD, &info);
            sommaloc = sommaloc + tmp;
        }
        else
        {
            sendTo = menum - dist;
            recvBy = menum - dist;
            tag = sendTo;

            MPI_Send(&sommaloc, 1, MPI_INT, sendTo, tag, MPI_COMM_WORLD);
            MPI_Recv(&tmp, 1, MPI_INT, recvBy, tag, MPI_COMM_WORLD, &info);
            sommaloc = sommaloc + tmp;
        }
    }

	MPI_Barrier(MPI_COMM_WORLD);
	T_fine = MPI_Wtime() - T_inizio;

	MPI_Reduce(&T_fine, &T_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/* Print the results (with the butterfly, every process holds the total sum!) */
	if(menum == 0)
	{
		printf("\nProcessors used (Strategy III - butterfly): %d\n", nproc);
		printf("\nTotal sum (in P0): %d\n", sommaloc);
		printf("\nMaximum execution time: %f\n", T_max);
	}
	else
	{
		/* Optional: uncomment to check that the other processes also hold the sum */
		/* printf("Process %d holds the total sum: %d\n", menum, sommaloc); */
	}

	MPI_Finalize();
	return 0;
}