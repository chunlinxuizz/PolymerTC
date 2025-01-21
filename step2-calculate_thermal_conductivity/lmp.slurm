#!/bin/bash
#SBATCH --job-name=pbttt_7x10x2i
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --output=%x.log
#SBATCH --partition=wzhcnormal

# load the environment
module purge
module load apps/lammps/23Jun2022/intel-21
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so

mpirun -np 64 lmp_mpi < in.nemd

