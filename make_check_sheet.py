"""
Make ONE picture sheet of random "check" extras, so you can judge many at once by eye.

Each tile = the image cut around one extra (orange box) + a number, the image id, type and score.
You write down which numbers are REAL extra objects. Then:
  real share = (real numbers) / (tiles)  -> tells us if the check rule removes noise or real extras.

Run from eval/MIG_Bench:
  python make_check_sheet.py --category outside --n 18     confirmed extras outside the boxes
  python make_check_sheet.py --category inside  --n 18     confirmed extras inside a box
  python make_check_sheet.py --category check   --n 18     "check" (looks like another type / pairing doubt)
  python make_check_sheet.py --category lowconf --n 18     low-confidence extras
  python make_check_sheet.py --category small   --n 18     small extras
"""

TRUE = "True"


def category_of(r):
    """Same order as the counter: check -> low-confidence -> small -> confirmed inside/outside."""
    if not r["status"].startswith("extra"):
        return "wanted"
    if r["check_reasons"]:
        return "check"
    if r.get("low_conf") == TRUE:
        return "lowconf"
    if r.get("small") == TRUE:
        return "small"
    return "inside" if r["status"] == "extra_inside" else "outside"
import argparse
import ast
import csv
import os
import random

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_csv", default="../result/extras_baseline_v3/per_detection.csv")
    ap.add_argument("--gen_dir", default="../result/coco_position/")
    ap.add_argument("--category", default="outside", choices=["outside", "inside", "check", "lowconf", "small"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.det_csv)) if category_of(r) == args.category]
    random.Random(args.seed).shuffle(rows)       # fixed seed -> same sample every time
    rows = rows[: args.n]
    print(f"{len(rows)} tiles (category '{args.category}')")

    tile, cols, label_h = 200, 6, 30
    sheet = Image.new("RGB", (cols * tile, ((len(rows) + cols - 1) // cols) * (tile + label_h)), "white")
    lines = []
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(args.gen_dir, f"bagel_{r['id']}.jpg")).convert("RGB").resize((512, 512))
        x1, y1, x2, y2 = ast.literal_eval(r["box"])
        d = ImageDraw.Draw(img)
        d.rectangle([x1, y1, x2, y2], outline=(240, 140, 0), width=3)
        # cut a square around the box with some margin, so you see the context
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) / 2 + 40
        crop = img.crop((max(0, cx - half), max(0, cy - half), min(512, cx + half), min(512, cy + half)))
        crop = crop.resize((tile, tile))
        px, py = (i % cols) * tile, (i // cols) * (tile + label_h)
        sheet.paste(crop, (px, py))
        ImageDraw.Draw(sheet).text((px + 4, py + tile + 4),
                                   f"#{i+1}  {r['id']}  {r['type']}  {float(r['score']):.2f}", fill=(0, 0, 0))
        lines.append(f"#{i+1}\tbagel_{r['id']}\t{r['type']}\tscore {r['score']}\t{r['status']}\t{r['check_reasons']}")

    out = args.out or os.path.join(os.path.dirname(args.det_csv), f"check_sheet_{args.category}.png")
    sheet.save(out)
    with open(out.replace(".png", ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved", out)


if __name__ == "__main__":
    main()
