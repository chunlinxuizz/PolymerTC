#!/bin/bash
#SBATCH -J gmx
#SBATCH -p wzhcnormal
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=64

module purge
module load apps/gromacs/2023.2-gcc9.3-intelmpi

gro_file=init-box.gro

is_normal_end(){
_=`grep 'Finished mdrun' $1.log`
if [ $? -ne 0 ]; then
    mv $1 $1_Failed
    exit 1
fi
}

job=em
mkdir -p $job
gmx_mpi grompp -f input_files/${job}.mdp -p force_field/system.top -c $gro_file -o ${job}.tpr -quiet -nocopyright -maxwarn 1 -nobackup -norenum
srun gmx_mpi mdrun -nobackup -deffnm $job
is_normal_end $job
#echo 0 | gmx_mpi trjconv -f ${job}.gro -s ${job}.tpr -pbc atom -o ${job}_pbcatom.gro
gro_file=${job}.gro
rm -r $job

job=npt-ann
mkdir -p $job
gmx_mpi grompp -f input_files/${job}.mdp -p  force_field/system.top -c $gro_file -o ${job}.tpr -quiet -nocopyright -maxwarn 1 -nobackup -norenum
srun gmx_mpi mdrun -nobackup -deffnm $job
is_normal_end $job
#echo 0 | gmx_mpi trjconv -f ${job}.gro -s ${job}.tpr -pbc atom -o ${job}_pbcatom.gro
gro_file=${job}.gro
rm -r $job

job=npt-equil
mkdir -p $job
gmx_mpi grompp -f input_files/${job}.mdp -p  force_field/system.top -c $gro_file -o ${job}.tpr -quiet -nocopyright -nobackup -norenum
srun gmx_mpi mdrun -nobackup -deffnm $job
is_normal_end $job
#echo 0 | gmx_mpi trjconv -f ${job}.gro -s ${job}.tpr -pbc atom -o ${job}_pbcatom.gro
gro_file=${job}.gro
rm -r $job

touch Finished
