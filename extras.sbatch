#!/bin/bash
#SBATCH -A kurs00110
#SBATCH -p kurs00110
#SBATCH --reservation=kurs00110
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH -t 02:00:00
#SBATCH -J cc-extras
#SBATCH -o cc-extras-%j.out

# usage: sbatch extras.sbatch            -> all 800 images
#        sbatch extras.sbatch 182,202,208,211  -> only these test images
source ~/miniforge3/bin/activate mig
module load cuda/12.9.1-rccb
export HF_HOME=$HPC_SCRATCH/hf
export HF_HUB_OFFLINE=1
cd ~/ConsistCompose/eval/MIG_Bench
IDS_ARG=""; OUT=../result/extras_baseline/
if [ -n "$1" ]; then IDS_ARG="--ids $1"; OUT=../result/extras_test/; fi
python count_extras.py --coco_path ../data/coco_position.jsonl \
  --gen_dir ../result/coco_position/ --out_dir $OUT $IDS_ARG
