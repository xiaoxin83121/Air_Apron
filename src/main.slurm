#!/bin/bash
#SBATCH -J main_zc
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH -o main.out

module load anaconda3
source activate
conda deactivate
conda activate zcenv
python /gs/home/tongchao/zc/Air_Apron/src/main.py ctdet --exp_id coco_dla --batch_size 5 --lr 1.25e-4  --gpus 0,1,2,3
conda deactivate