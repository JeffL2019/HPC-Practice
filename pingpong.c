/*
 * pingpong.c - measure message latency and bandwidth between two MPI ranks.
 *
 * Rank 0 sends a message to rank 1, and rank 1 sends it straight back.
 * Timing many of these round trips tells us:
 *   - latency: how long one small message takes to arrive
 *   - bandwidth: how much data per second the network can move
 */

 #include <mpi.h> // MPI functions and types (MPI_Init, MPI_Send, MPI_Recv)
 #include <stdio.h> // printf
 #include <stdlib.h> // malloc, free
 #include <string.h> // memset

#define MAX_BYTES (4 * 1024 * 1024) // max message size we test: 4MB
#define WARMUP 10 // number of warmup iterations before measuring

int main(int argc, char **argv) {
    // Start MPI
    MPI_Init(&argc, &argv);

    int rank, size;

    // MPI_COMM_WORLD is the communicator. the group containing every process
    // rank is this process's rank within the communicator (0, 1, ...)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // size is the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // check if there are 2 and only 2 processes
    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Run with exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    // which machine am I on?
    char host[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(host, &len);
    printf("Hello from rank %d out of %d on %s\n", rank, size, host);
    fflush(stdout);

    // Create message buffer of max bytes. Both rank need one.
    // Sender needs a buffer to read from
    // Writer needs a buffer to write to
    char *buf = malloc(MAX_BYTES);
    memset(buf, 'a', MAX_BYTES);

    // create a barrier so what every rank waits until all ranks arrive.
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n%12s %8s %14s %18s\n",
               "Bytes", "Iters", "Latency (us)", "Bandwidth (MB/s)");
    }

    // Start ping pong
    // Test message size from 1B, 2B, 4B, ..., up to 4MB
    for (int bytes = 1; bytes <= MAX_BYTES; bytes *= 2) {
        int iters = (bytes <= 65536) ? 10000 : 50;

        // create barrier so all ranks are ready
        MPI_Barrier(MPI_COMM_WORLD);

        double start = 0.0;
        for (int i = 0; i < WARMUP + iters; i++) {
            // Start the clock only after the warmup round trips
            if (i == WARMUP) start = MPI_Wtime();

            if (rank == 0) {
                // PING: send `bytes` chars from buf to rank 1.
                // Args: buffer, count, datatype, destination rank, tag, communicator.
                // The tag (0) is a label the receiver can use to match messages.
                MPI_Send(buf, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

                // Wait for rank 1 to send it back.
                // Args: buffer, count, datatype, source rank, tag, communicator, status.
                // MPI_STATUS_IGNORE = we don't need details about the message.
                MPI_Recv(buf, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // PONG: wait for rank 0's message, then send it right back.
                // Order is reversed (receive first, then send).
                // If both ranks called MPI_Send first, result would be a deadlock
                MPI_Recv(buf, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
        // MPI_Wtime returns wall-clock time in seconds
        double elapsed = MPI_Wtime() - start;

        if (rank == 0) {
            // Each iteration is a round trip = two one-way messages.
            double one_way = elapsed / (2.0 * iters); // seconds per message
            double latency_us = one_way * 1e6; // microseconds
            double bandwidth = bytes / one_way / 1e6; // megabytes per second
            printf("%12d %8d %14.2f %18.2f\n", bytes, iters, latency_us, bandwidth);
        }
    }

    free(buf);
    // Shut down MPI cleanly
    MPI_Finalize();
    return 0;
}