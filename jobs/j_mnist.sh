#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=SVDD-M
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Deep-SVDD-PyTorch/out-mnist.log


mkdir log/mnist

cd src

python main.py mnist mnist_LeNet ../log/mnist ../data --objective one-class --lr 0.0001 --n_epochs 150 \
    --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True \
    --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 \
    --ae_weight_decay 0.5e-3 \
    --normal_class "0, 1, 2, 3, 4, 5, 6, 7"