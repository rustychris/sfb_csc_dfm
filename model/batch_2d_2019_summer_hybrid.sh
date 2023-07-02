#!/bin/bash -l
#SBATCH --job-name sfb_csc
#SBATCH -o slurm_out-%j.output
#SBATCH -e slurm_out-%j.output
#SBATCH --partition high2
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --time 10-00:00:00

conda activate general
. prepenv.sh

# . /share/apps/intel-2019/bin/compilervars.sh intel64
# PREFIX=/home/rustyh/src/dfm/t140737
export DFM_ROOT=$CONDA_PREFIX
# Trying to use DELFT_SRC to get share path, so it can find proc_def
export DELFT_SRC=
# export PATH=$PREFIX/bin:$PATH
# export LD_LIBRARY_PATH=$DFM_ROOT/lib:$PREFIX/lib:$LD_LIBRARY_PATH

python -u hybrid_model.py --run-dir data_2d_2019_hybrid -n 16 -l 0 -p 2019-04-01:2019-04-02 --salinity
