#!/bin/bash
#PBS -N gene_distributed_mgr_opt
#PBS -l nodes=1366:ppn=24
#PBS -l walltime=00:30:00
# #PBS -q test
#PBS -j oe

# Change to the direcotry that the job was submitted from
cd $PBS_O_WORKDIR

# Launch the parallel job to the allocated compute nodes
#aprun -n 1024 -N 16 -cc depth -j1 ./gene_hazel_hen > fg_nlitgae_1.out 2>&1

./run.sh > opt.out 2>&1
