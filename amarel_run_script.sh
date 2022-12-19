#!/bin/bash
#SBATCH --partition=main            # Partition (job queue)
#SBATCH --requeue                   # Return job to the queue if preempted
#SBATCH --job-name=minstMPa         # Assign a short name to your job
#SBATCH --nodes=1                   # Number of nodes you require
#SBATCH --ntasks=1                 # Total # of tasks across all nodes
#SBATCH --cpus-per-task=5           # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                # Real memory (RAM) required (MB)
#SBATCH --time=48:00:00             # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out    # STDOUT output file
#SBATCH --error=slurm.%N.%j.err     # STDERR output file (optional)

module load intel/19.0.3 mvapich2/2.2
srun --mpi=pmi2 python minst_test_diego_mp.py