#!/bin/bash -l
#SBATCH --job-name sfb_csc
#SBATCH -o slurm_out.output
#SBATCH -e slurm_out.output
#SBATCH --partition high2
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --time 10-00:00:00

# used to have -%j in output names

conda activate dfm_t142431
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# . /share/apps/intel-2019/bin/compilervars.sh intel64
# PREFIX=/home/rustyh/src/dfm/t140737
export DFM_ROOT=$CONDA_PREFIX
# Trying to use DELFT_SRC to get share path, so it can find proc_def
# export DELFT_SRC=$CONDA_PREFIX/build/dfm/
# export PATH=$PREFIX/bin:$PATH
# export LD_LIBRARY_PATH=$DFM_ROOT/lib:$PREFIX/lib:$LD_LIBRARY_PATH

python -u hybrid_model.py --run-dir data_2d_2018_hybrid -n 16 -l 0 -p 2018-04-01:2018-11-01 --salinity

# sample direct call:
#srun -n 16 --mpi=pmi2 -o job-hybrid-%2t.out python /home/rustyh/src/sfb_csc_dfm/model/hybrid_model.py --bmi --mpi=slurm --mdu data_2d_2019_hybrid-v003/flowfm.mdu

