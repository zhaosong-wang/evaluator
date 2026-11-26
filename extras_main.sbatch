#!/bin/bash
#SBATCH -A kurs00110
#SBATCH -p kurs00110
#SBATCH --reservation=kurs00110
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH -t 02:00:00
#SBATCH -J cc-extras-main
#SBATCH -o /work/scratch/kurse/kurs00110/%u/cc-extras-main-%j.out
# usage: sbatch extras_main.sbatch <image_dir> <name> [ids]  -> $S/extras_<name>/
S="/work/scratch/kurse/kurs00110/${USER}"
source ~/miniforge3/bin/activate mig
module load cuda/12.9.1-rccb
export HF_HOME=$S/hf HF_HUB_OFFLINE=1 TQDM_DISABLE=1
cd ~/ConsistCompose/eval/MIG_Bench
GEN=${1:-../result/coco_position/}; OUT=$S/extras_${2:-baseline}/; IDS_ARG=""
if [ -n "$3" ]; then IDS_ARG="--ids $3"; fi
python count_extras.py --coco_path ../data/coco_position.jsonl --gen_dir $GEN \
  --out_dir $OUT $IDS_ARG --detect per_type 2> $S/extras_${2:-baseline}_err.txt
