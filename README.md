# Extra-object evaluator v7 (Topic 1, Group 6)

Counts extra objects in generated COCO-Position images, on top of the official MIG-Bench score.
Same GroundingDINO and settings as MIG-Bench.

## Setup
1. Put the 4 files into `ConsistCompose/eval/MIG_Bench/`.
2. Activate the env you use for MIG-Bench scoring, then run `python -m pytest test_count_extras.py -q` (expect 29 passed).
3. In `extras_main.sbatch`, change `S=` and the `#SBATCH -o` path to your own scratch folder, and the `source ... activate` line to your env.

## Run
    sbatch extras_main.sbatch <image_dir> <name>          # all 800 images, ~15 min
    sbatch extras_main.sbatch <image_dir> <name> 202,208  # quick test
Results: `$S/extras_<name>/summary.json` (starts with `main_metric`), `per_image.csv`, `per_detection.csv`, `vis/`.
The log ends with one `MAIN METRIC` line.

## Main metric
**extra_outside per image** = confirmed extras of the requested types, outside all their boxes. Baseline: **0.537** (lower is better).
Also: precision_outside = wanted / (wanted + extra_outside) (baseline 0.876), recall = wanted / requested (baseline 0.953, must not drop).
mIoU and AP from the official MIG-Bench score should stay about the same (baseline 85.17 / 70.96).

## Labels (R = requested box, d = detection, s = score)
Pairing (one d per R, highest IoU first): try 1 IoU(R,d) >= 0.5; try 2 d mostly inside R; try 3 d covers most of R -> wanted. R without pair -> missing.
Leftover d, first match wins: check (looks like another requested object, not counted) -> same_object (covers an already paired R, not counted)
-> lowconf (s < 0.35) -> small (< 1% of the image) -> extra_inside (mostly inside an R of its type) -> extra_outside.

## Manual check
    python check_sheet_v6.py --per_det $S/extras_<name>/per_detection.csv --coco_path ../data/coco_position.jsonl \
        --gen_dir <image_dir> --out_dir $S/check_<name>/ --status extra_outside --n 20
Label each tile: A real extra, B same object / part / group box, C wrong type / reflection / nothing.
