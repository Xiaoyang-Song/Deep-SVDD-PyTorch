#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=SVDD-MFM32
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Deep-SVDD-PyTorch/out/out-mnist-fashionmnist-32.log


mkdir log/mnist-fashionmnist-32

cd src

python main.py "mnist-fashionmnist-32" cifar10_LeNet "../log/mnist-fashionmnist-32" ../data \
    --objective one-class --lr 0.0001 --n_epochs 150 \
    --lr_milestone 50 --batch_size 200 \
    --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 \
    --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
    --normal_class "0" --experiment_type "between"
    