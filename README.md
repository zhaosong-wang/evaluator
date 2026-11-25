# Extra-object evaluator (Topic 1, Group 6)

Counts **extra objects**: objects the model drew in addition to the requested ones.
It uses the same GroundingDINO and settings as MIG-Bench (box threshold 0.25, NMS 0.5).
MIG-Bench keeps only the best detection per requested box. This tool labels all the other detections.

Rule design: Ilyas. Code and tests: written with Claude (AI assistant) from that rule.

## The rule (per object type, e.g. "chair")
R = requested box from the prompt, d = one GroundingDINO detection.

1. **Pair one d per R (one-to-one)**
   - Try 1: IoU(R, d) >= 0.5, highest IoU first -> `wanted`
   - Try 2 (object drawn too small): IoA(d, R) >= 0.5 -> `wanted_try2`
   - R with no pair -> `missing`
2. **Label each unpaired d (first match wins)**
   1. IoU >= 0.5 with a requested box of ANOTHER type, or inside R with a higher score than R's pair -> `check` (not counted)
   2. score < 0.35 -> `extra_lowconf`
   3. area < 1% of the image -> `extra_small`
   4. IoA(d, R) >= 0.5 -> `extra_inside`, else `extra_outside`

IoU(R, d) = overlap / (R + d - overlap). IoA(d, R) = overlap / area of d (order matters).

**Metrics** (summary.json):
- recall = wanted / requested
- precision_confirmed = wanted / (wanted + extra_inside + extra_outside)
- precision_all = wanted / (wanted + extras + lowconf + small)

Limit: a lower bound. NMS merges stacked objects of one type, and doubtful cases go to `check`.

## Setup (Lichtenberg)
Needs the MIG-Bench env (`mig`) with GroundingDINO working (see MIG-Bench setup).
Copy all files into `ConsistCompose/eval/MIG_Bench/`, then:

```bash
conda activate mig
python -m pytest test_count_extras.py -q     # expect: 15 passed
```

## Run
```bash
sbatch extras.sbatch                   # all 800 COCO-Position images (GPU, ~15 min)
sbatch extras.sbatch 182,202,208,211   # only these images (quick test)
```
Edit paths in `extras.sbatch` if your env or folders differ (`source ~/miniforge3/bin/activate mig`).

Re-apply the rules without GPU (after changing thresholds):
```bash
python count_extras.py --coco_path ../data/coco_position.jsonl --gen_dir ../result/coco_position/ \
  --out_dir ../result/extras_new/ --from_csv ../result/extras_baseline/per_detection.csv --no_draw
```

Outputs in `--out_dir`: `summary.json`, `per_image.csv`, `per_detection.csv`, `review_list.csv`, `vis/`.

## Helpers
- `draw_examples.py --ids 202,208` : meeting pictures with the exact GroundingDINO boxes (T = requested, E = extra)
- `make_check_sheet.py --category outside|inside|check|lowconf|small --n 30` : random sample for manual checking
- `sort_by_category.py` : sorts images into folders (clean, outside_extra, inside_extra, ...)

## Baseline result (ConsistCompose, 800 images)
0.76 confirmed extras / image (0.65 outside, 0.11 inside); 39% of images have >= 1; 29% fully clean.
recall 0.93, precision_confirmed 0.83, precision_all 0.68.
