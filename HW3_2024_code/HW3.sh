#! /bin/bash

#SBATCH --clusters=htc
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --partition=short
#SBATCH --array=1-4

cd $SCRATCH
module load Anaconda3/2022.05
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
module load torchvision/0.11.3-foss-2021a-CUDA-11.3.1

rsync $DATA/Deep_Learning_HW3/HW3_main.py ./
rsync $DATA/Deep_Learning_HW3/HW3_methods.py ./
config=$DATA/Deep_Learning_HW3/config.txt

lr=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
dropout=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
n_channels=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

python -u HW3_main.py --lr ${lr} --dropout ${dropout} --n_channels ${n_channels} --num_epochs=100