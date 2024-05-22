#!/bin/bash -l
##This is a temporary test shell file


# Set SLURM output file
#SBATCH --output=./submittest/%j.out
#SBATCH --job-name=ivandl:submittest
#SBATCH --partition=a100-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:20  # time (D-HH:MM)

echo "running with test work"

conda activate venv
module load cuda/11.8
srun python seg.py $SLURM_JOB_ID


if [ $? -eq 0 ]; then
    echo "ended successfully"
else
    echo "Task failed or ended abnormally"
fi