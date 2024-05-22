#!/bin/bash -l
#
# Set SLURM output file
#SBATCH --output=./result/setr/%jt2.out
#SBATCH --job-name=ivandl:setr
#SBATCH --partition=a100
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
# srun python ./tools/train.py ./configs/setr/setr_voc.py --work-dir ./result
# srun python ./tools/train.py ./configs/setr/setr_uq6.py --work-dir ./result/setr/uq
# srun python ./tools/train.py ./configs/setr/setr_uq6t2.py --work-dir ./result/setr/uq
# srun python ./tools/train.py ./configs/setr/setr_uq12t2.py --work-dir ./result/setr/uq
srun python ./tools/train.py ./configs/setr/setr_uq12.py --work-dir ./result/setr/uq
# srun python ./tools/train.py ./configs/setr/setr_uq30t2.py --work-dir ./result/setr/uq
# srun python ./tools/train.py ./configs/setr/setr_uq30.py --work-dir ./result/setr/uq
if [ $? -eq 0 ]; then
    echo "ended successfully"
else
    echo "Task failed or ended abnormally"
fi