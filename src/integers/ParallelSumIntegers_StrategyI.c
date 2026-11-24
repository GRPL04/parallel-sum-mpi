/* --------------------------------------------------------------------------
  PURPOSE
  Computation of the sum of n integers in parallel using
  Strategy I (sequential collection of the partial sums in process 0).
--------------------------------------------------------------------------
  DESCRIPTION
  First, process zero reads the input data and distributes them
  among the processors. Each processor computes its partial sum (sommaloc).
  Then, applying Strategy I, each processor with id != 0
  sends its partial sum to processor 0, which receives them one after another
  in a loop and accumulates the total.
----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main (int argc, char **argv)
{
	/* Variable declarations */
    int menum, nproc, tag;
	int n, nloc, i, resto, nlocgen;
	int ind, tmp;
	int *vett, *vett_loc;
	int sommaloc = 0;
	double T_inizio, T_fine, T_max;

	MPI_Status info;

	/* Initialise the MPI environment */
	MPI_Init(&argc, &argv);
	/* Store the process ID in menum */
	MPI_Comm_rank(MPI_COMM_WORLD, &menum);
	/* Store the number of processors in nproc */
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	/* Read the input data */
	if (menum == 0)
	{
		printf("Enter the number of elements to add: \n");
		fflush(stdout);
		scanf("%d", &n);

       	vett = (int*)calloc(n, sizeof(int));
	}

	/* Send n to all the processors in MPI_COMM_WORLD */
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Number of addends assigned to each processor */
	nlocgen = n / nproc; // integer division
    resto = n % nproc; // remainder of the division

	/* If the remainder is not zero, the first "resto" processes get one extra addend */
	if (menum < resto)
	{
		nloc = nlocgen + 1;
	}
	else
	{
		nloc = nlocgen;
	}

    /* Memory allocation for the local vector */
	vett_loc = (int*)calloc(nloc, sizeof(int));

	// P0 generates and assigns the elements to add
	if (menum == 0)
	{
        /* Seed the random generator with the current system time */
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

		/* Assign the first addends to P0 */
        for(i = 0; i < nloc; i++)
		{
			*(vett_loc + i) = *(vett + i);
		}

		/* ind is the number of addends already assigned */
		ind = nloc;

		/* P0 assigns the remaining addends to the other processors */
		for(i = 1; i < nproc; i++)
		{
			tag = i; /* Message tag equal to the ID of the receiving process */
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
		/* Receiving phase in the other processors */
		MPI_Recv(vett_loc, nloc, MPI_INT, 0, tag, MPI_COMM_WORLD, &info);
	}

	/* Synchronise the processors of MPI_COMM_WORLD */
	MPI_Barrier(MPI_COMM_WORLD);

	T_inizio = MPI_Wtime(); // Start the timer

	/* Each processor computes its local partial sum */
	for(i = 0; i < nloc; i++)
	{
		sommaloc = sommaloc + *(vett_loc + i);
	}

	/* --- STRATEGY 1 (SEQUENTIAL COLLECTION IN PROCESS 0) --- */
	if (menum != 0)
	{
		/* Processes other than 0 send their partial sum to process 0 */
		tag = menum;
		MPI_Send(&sommaloc, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
	}
	else
	{
		/* Process 0 receives the partial sums of the other processes one by one and adds them up */
		for(i = 1; i < nproc; i++)
		{
			tag = i;
			MPI_Recv(&tmp, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &info);
			sommaloc = sommaloc + tmp;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD); // Synchronisation
	T_fine = MPI_Wtime() - T_inizio; // Elapsed time

	/* Maximum execution time over all processes */
	MPI_Reduce(&T_fine, &T_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/* Print the final results (process 0 only) */
	if(menum == 0)
	{
		printf("\nProcessors used (Strategy I): %d\n", nproc);
		printf("\nThe total sum is: %d\n", sommaloc);
		printf("\nLocal computation time (P0): %lf\n", T_fine);
		printf("\nMaximum time (MPI_Reduce): %f\n", T_max);
	}

	/* Shut down the MPI environment */
	MPI_Finalize();
	return 0;
}