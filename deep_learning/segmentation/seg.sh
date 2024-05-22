#!/bin/bash -l
#
# Set SLURM output file
#SBATCH --output=./fcn/%j.out
#SBATCH --job-name=ivandl:fcn
#SBATCH --partition=a100
#SBATCH --time=6-24:00  # time (D-HH:MM)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END


echo "running with deep learning work"

conda activate venv
module load cuda/11.8
srun python run.py $SLURM_JOB_ID
if [ $? -eq 0 ]; then
    echo "ended successfully"
else
    echo "Task failed or ended abnormally"
fi