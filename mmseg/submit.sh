#!/bin/bash -l
#
# Set SLURM output file
#SBATCH --output=./result/%j.out
#SBATCH --job-name=ivandl:result
#SBATCH --partition=p100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-24:00  # time (D-HH:MM)
#SBATCH --mail-type=END



echo "running with deep learning work"

conda activate venv
module load cuda/11.8
srun python ./tools/train.py ./configs/segformer/seg_uq.py --work-dir ./result
if [ $? -eq 0 ]; then
    echo "ended successfully"
else
    echo "Task failed or ended abnormally"
fi