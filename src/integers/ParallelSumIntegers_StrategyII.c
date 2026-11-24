/* --------------------------------------------------------------------------
  PURPOSE
  Computation of the sum of n integers
  on a parallel computer of type MIMD
  with p processors, where p is a power of 2.
--------------------------------------------------------------------------
  DESCRIPTION
  First, process zero reads the input data and distributes them
  among the processors, so that each of them has the same number
  of elements, give or take one.
  These elements are stored in a vector x of
  size nloc (an integer that depends on the processor).
  Then each processor computes its partial sum and sends it to another processor,
  so that the total sum of the input elements is reached step by step.
  Using MPI_Wtime, each processor measures the
  execution time of the sum. Processor 0 receives these
  times and takes their maximum as the execution time of the whole sum.
  The number of processes must be equal to 2^p with p>=1.
  The number of elements to add must be greater than or equal to the number
  of processors.
----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------
  Header files used
----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*--------------------------------------------------------------------------
  Header file with the preprocessor definitions needed
  to use MPI.
-----------------------------------------------------------------------------*/
#include <mpi.h>

int main (int argc, char **argv)
{
	/* variable declarations */
    int menum, nproc, tag;
	int n, nloc, i, somma, resto, nlocgen;
	int ind, p, r, sendTo, recvBy, tmp;
	int *potenze, *vett, *vett_loc, passi=0;
	int sommaloc=0;
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

   		// Print the vector to be added if it has fewer than 100 elements
		if (n < 100)
		{
			for (i = 0; i < n; i++)
			{
				printf("\n\nVector element %d = %d ", i, *(vett + i));
			}
        }

		// Assign the first addends to P0
        for(i = 0; i < nloc; i++)
		{
			*(vett_loc + i) = *(vett + i);
		}

  		// ind is the number of addends already assigned
		ind = nloc;

		/* P0 assigns the remaining addends to the other processors */
		for(i = 1; i < nproc; i++)
		{
			tag = i; /* Message tag equal to the ID of the receiving process */
			/* IF there are leftover addends to share among the processors */
            if (i < resto)
			{
				/* P0 sends it the corresponding local vector, with one extra addend */
				MPI_Send(vett + ind, nloc, MPI_INT, i, tag, MPI_COMM_WORLD);
				ind = ind + nloc;
			}
			else
			{
				/* P0 sends it the corresponding local vector */
				MPI_Send(vett + ind, nlocgen, MPI_INT, i, tag, MPI_COMM_WORLD);
				ind = ind + nlocgen;
			}// end if
		}// end for
	}

    /* IF this is not P0, receive the data sent by P0 */
    else
    {
		// tag equals the number of the receiving processor
		tag = menum;

		/* Receiving phase */
		MPI_Recv(vett_loc, nloc, MPI_INT, 0, tag, MPI_COMM_WORLD, &info);
	}// end if

	/* Synchronise the processors of MPI_COMM_WORLD */
	MPI_Barrier(MPI_COMM_WORLD);

	T_inizio = MPI_Wtime(); // Start the timer

	for(i = 0; i < nloc; i++)
	{
		/* Each processor computes its partial sum */
		sommaloc = sommaloc + *(vett_loc + i);
	}

	/* --- STRATEGY 2 (BINARY TREE TOWARDS P0) --- */
	// Compute p = log_2 (nproc)
	p = nproc;

	while(p != 1)
	{
		/* Shift one bit to the right */
		p = p >> 1;
		passi++;
	}

	/* Vector with the powers of 2 */
	potenze = (int*)calloc(passi + 1, sizeof(int));

	for(i = 0; i <= passi; i++)
	{
		potenze[i] = p << i;
	}

	/* Partial sums and combination of the partial results */
	for(i = 0; i < passi; i++)
	{
		// ... Identify the role of this processor
		r = menum % potenze[i + 1];

		// If the remainder is 2^i, processor menum sends
		if(r == potenze[i])
		{
			// ID of the processor that receives the local sum
			sendTo = menum - potenze[i];
			tag = sendTo;
			MPI_Send(&sommaloc, 1, MPI_INT, sendTo, tag, MPI_COMM_WORLD);
		}
		else if(r == 0) // If the remainder is 0, processor menum receives
		{
			recvBy = menum + potenze[i];
			tag = menum;
			MPI_Recv(&tmp, 1, MPI_INT, recvBy, tag, MPI_COMM_WORLD, &info);
			/* Partial sum at step i */
			sommaloc = sommaloc + tmp;
		}// end
	}// end for

	MPI_Barrier(MPI_COMM_WORLD); // Synchronisation
	T_fine = MPI_Wtime() - T_inizio; // Elapsed time

	/* Total execution time */
	MPI_Reduce(&T_fine, &T_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/* Print the final results */
	if(menum == 0)
	{
		printf("\nProcessors used: %d\n", nproc);
		printf("\nThe sum is: %d\n", sommaloc);
		printf("\nLocal computation time: %lf\n", T_fine);
		printf("\nMaximum time MPI_Reduce: %f\n", T_max);
	}// end if

	/* Shut down the MPI environment */
	MPI_Finalize();
}// end of program