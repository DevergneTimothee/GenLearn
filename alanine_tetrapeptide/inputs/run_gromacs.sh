#!/bin/bash


export OMP_NUM_THREADS=1


gmx_mpi grompp -f md.mdp -c ala4.49000.gro -p ala4.amber99sb-tip3p.top -o input.tpr

mpiexec -n 1 gmx_mpi mdrun -s input.tpr -deffnm ala4 -plumed plumed.dat -nsteps 10000000 -pin on -pinoffset 44 -gpu_id 0

