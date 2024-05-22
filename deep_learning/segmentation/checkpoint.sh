#!/bin/bash -l
#
# Set SLURM output file
#SBATCH --output=./checkpoint/%j.out
#SBATCH --job-name=ivandl:checkpoint
#SBATCH --partition=a100-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:20  # time (D-HH:MM)

echo "running with  test work"

conda activate venv
module load cuda/11.8
srun python checkpoint.py


if [ $? -eq 0 ]; then
    echo "ended successfully"
else
    echo "Task failed or ended abnormally"
fi