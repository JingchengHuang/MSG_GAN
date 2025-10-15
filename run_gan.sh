#!/bin/bash
# >>> TIME >>>
# 2025-10-15
# <<< TIME <<<
#SBATCH --job-name=MSG_GAN
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --comment=gan

source ~/.bashrc
module load app/cuda/11.8
conda activate hjc

CUDA_CHECK_SCRIPT=$(cat << 'EOF'
import torch
print(f"PyTorch 版本:{torch.__version__}")
print(f"CUDA 可用:{torch.cuda.is_available()}")
EOF
)

echo "==CUDA CHECK$(date)=="
time python -c "$CUDA_CHECK_SCRIPT"
echo "==JOB START$(date)=="
cd ~/hjc_files/MSG_GAN/

python 1.0_train_FiveStamp.py
