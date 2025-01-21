#!/bin/bash
#SBATCH --job-name=btt_35x10x2
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --output=%x.log
#SBATCH --partition=wzhcnormal
#SBATCH --exclude=g02r1n06

# load the environment
module purge
export PATH=$PATH:/work/home/acuoiis6xe/.conda/envs/lammps/bin
module load apps/gromacs/2022.2/hpcx-gcc-7.3.1
export PATH=$PATH:/work/home/acuoiis6xe/apps/lammps-2Aug2023/build
export LD_LIBRARY_PATH=/work/home/acuoiis6xe/apps/lammps-2Aug2023/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/home/acuoiis6xe/apps/lammps-2Aug2023/src:$LD_LIBRARY_PATH

time python dsf_repeat.py 
