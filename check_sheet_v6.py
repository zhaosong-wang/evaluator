"""Manual-check sheets for one counter label (v6).

Picks N random detections with one status (default: confirmed extra_outside) and draws numbered
tiles: left = whole image with all boxes, right = zoom on the detection.
  red    = requested boxes of the same type
  green  = wanted detections of the same type
  purple = other extras of the same type in this image (context only, judged in their own tiles)
  orange = THE detection to judge
Writes check_<status>_<k>.png sheets + check_<status>_key.csv (tile -> image id, type, box, score).

Usage (from eval/MIG_Bench, conda env mig):
  python check_sheet_v6.py --per_det $HPC_SCRATCH/extras_v6_coco_gpu/per_detection.csv \
      --coco_path ../data/coco_position.jsonl --gen_dir ../result/coco_position/ \
      --out_dir $HPC_SCRATCH/check_v6/ --status extra_outside --n 40
Label each tile: A = real separate extra, B = same object / part / group box, C = wrong type, reflection or nothing.
"""
import argparse
import csv
import json
import os
import random
import re

from PIL import Image, ImageDraw, ImageFont

IMG = 512
TILE = 300          # each panel is TILE x TILE
HEAD = 92           # text lines above the panels (id + prompt)


def object_type(label):
    return re.sub(r"^(a|an|the)\s+", "", label.strip().lower())


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def confirmed(r):
    return r["low_conf"] != "True" and r.get("small") != "True" and not r["check_reasons"]


def tile(img, req, wanted, det, num, text, prompt="", others=()):
    """img: 512x512 PIL; req/wanted: boxes; det: box to judge."""
    full = img.copy()
    d = ImageDraw.Draw(full)
    for b in req:
        d.rectangle(b, outline=(230, 30, 30), width=3)
    for b in wanted:
        d.rectangle(b, outline=(30, 190, 70), width=3)
    for b in others:
        d.rectangle(b, outline=(160, 60, 220), width=2)
    d.rectangle(det, outline=(255, 150, 0), width=5)
    # zoom: detection box + 25% margin, at least 128 px, kept square inside the image
    x1, y1, x2, y2 = det
    side = max(x2 - x1, y2 - y1) * 1.25
    side = min(max(side, 128), IMG)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - side / 2, 0), IMG - side)
    top = min(max(cy - side / 2, 0), IMG - side)
    zoom = full.crop((left, top, left + side, top + side)).resize((TILE, TILE))
    out = Image.new("RGB", (2 * TILE + 6, TILE + HEAD), "white")
    out.paste(full.resize((TILE, TILE)), (0, HEAD))
    out.paste(zoom, (TILE + 6, HEAD))
    dr = ImageDraw.Draw(out)
    dr.text((6, 4), f"#{num}  {text}", fill="black", font=font(20))
    words, lines, line = prompt.split(), [], ""
    for wd in words:                       # wrap the prompt into up to 3 lines
        if len(line) + len(wd) + 1 > 68:
            lines.append(line); line = wd
        else:
            line = (line + " " + wd).strip()
    lines.append(line)
    if len(lines) > 3:
        lines = lines[:3]; lines[2] = lines[2][:64] + " ..."
    for i, ln in enumerate(lines):
        dr.text((6, 30 + 20 * i), ln, fill=(40, 40, 160), font=font(15))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_det", required=True)
    ap.add_argument("--coco_path", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--status", default="extra_outside",
                    help="extra_outside | extra_inside | same_object | wanted_try3 | extra_lowconf")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per_sheet", type=int, default=20)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.coco_path))
    rows = list(csv.DictReader(open(args.per_det)))
    by_img = {}
    for r in rows:
        r["box"] = json.loads(r["box"])
        by_img.setdefault(r["id"], []).append(r)

    if args.status == "extra_lowconf":
        pool = [r for r in rows if r["status"].startswith("extra") and r["low_conf"] == "True"
                and r.get("small") != "True" and not r["check_reasons"]]
    elif args.status.startswith("extra"):
        pool = [r for r in rows if r["status"] == args.status and confirmed(r)]
    else:
        pool = [r for r in rows if r["status"] == args.status]
    print(f"{len(pool)} detections with status {args.status}; sampling {min(args.n, len(pool))} (seed {args.seed})")
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))

    os.makedirs(args.out_dir, exist_ok=True)
    key_path = os.path.join(args.out_dir, f"check_{args.status}_key.csv")
    tiles = []
    with open(key_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "id", "type", "score", "box", "label", "prompt"])
        for k, r in enumerate(sample, 1):
            img = Image.open(os.path.join(args.gen_dir, f"bagel_{r['id']}.jpg")).convert("RGB").resize((IMG, IMG))
            req = [[v * IMG for v in s["bbox"]] for s in data[r["id"]]["segment"]
                   if object_type(s["label"]) == r["type"]]
            wanted = [o["box"] for o in by_img[r["id"]]
                      if o["type"] == r["type"] and o["status"].startswith("wanted") and o is not r]
            prompt = data[r["id"]].get("caption", "").strip()
            others = [o["box"] for o in by_img[r["id"]]
                      if o["type"] == r["type"] and o["status"].startswith("extra") and o is not r]
            tiles.append(tile(img, req, wanted, r["box"], k, f"img {r['id']}  {r['type']}  {float(r['score']):.2f}", prompt, others))
            w.writerow([k, r["id"], r["type"], r["score"], r["box"], "", prompt])

    tw, th = tiles[0].size if tiles else (0, 0)
    for s in range(0, len(tiles), args.per_sheet):
        chunk = tiles[s:s + args.per_sheet]
        rows_n = (len(chunk) + args.cols - 1) // args.cols
        sheet = Image.new("RGB", (args.cols * (tw + 10), rows_n * (th + 10)), (200, 200, 200))
        for i, t in enumerate(chunk):
            sheet.paste(t, ((i % args.cols) * (tw + 10), (i // args.cols) * (th + 10)))
        path = os.path.join(args.out_dir, f"check_{args.status}_{s // args.per_sheet + 1}.png")
        sheet.save(path)
        print("wrote", path)
    print("wrote", key_path)


if __name__ == "__main__":
    main()
