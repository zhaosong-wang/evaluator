"""
Draw meeting examples: counter results in the visual style of annotate_unwanted.py (groupmate's script).

  T1, T2 ...  blue dashed   = requested boxes (from the benchmark file)
  E1, E2 ...  red solid     = confirmed extras (counter: extra_outside / extra_inside)
  E1, E2 ...  amber dashed  = not sure (counter: low-confidence, small, or "check")

Boxes are GroundingDINO's exact boxes from per_detection.csv, only scaled from the 512x512 detector
image to the image's own size. Line width scales with box size, so tiny extras stay visible.
Wanted detections are not drawn (add --show_wanted to draw them thin green).

Run from eval/MIG_Bench (no GPU, plain Pillow):
  python draw_examples.py --ids 182,202,208,211,21,550
Writes ../result/extras_examples/bagel_<id>.jpg + bagel_<id>.txt (what each E tag is).
"""
import argparse
import ast
import csv
import json
import os

from PIL import Image, ImageDraw, ImageFont

DET_SIZE = 512                      # counter / GroundingDINO image size
BLUE, RED, AMBER, GREEN = (100, 181, 255), (204, 21, 39), (255, 188, 50), (40, 170, 90)
DARK = (16, 21, 28)


def font(size):
    for p in ["/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def line_width(box):
    """2 px for tiny boxes, up to 6 px for big ones (at 1024 px image size)."""
    side = min(box[2] - box[0], box[3] - box[1])
    return int(max(2, min(6, side / 15)))


def dashed_rect(d, box, color, width, dash=12, gap=7):
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]:
        length = max(abs(bx - ax), abs(by - ay))
        steps = int(length // (dash + gap)) + 1
        for i in range(steps):
            s = i * (dash + gap) / max(length, 1)
            e = min(1.0, (i * (dash + gap) + dash) / max(length, 1))
            d.line([(ax + (bx - ax) * s, ay + (by - ay) * s), (ax + (bx - ax) * e, ay + (by - ay) * e)],
                   fill=color, width=width)


def draw_box(d, box, tag, color, img_w, img_h, dashed=False, target=False):
    w = 3 if target else line_width(box)
    outline_box = [box[0], box[1], box[2], box[3]]
    # dark outline under the colored line, like the groupmate's style, so it shows on any background
    if dashed:
        dashed_rect(d, outline_box, DARK, w + 2)
        dashed_rect(d, outline_box, color, w)
    else:
        d.rectangle(outline_box, outline=DARK, width=w + 2)
        d.rectangle(outline_box, outline=color, width=w)
    size = 22 if target else max(20, min(30, int((box[2] - box[0]) / 3)))
    f = font(int(size * 0.62))
    lx = max(2, min(box[0], img_w - size - 2))
    ly = min(img_h - size - 2, box[1] + 2) if target else max(2, box[1] - size - 2)
    d.rounded_rectangle([lx, ly, lx + size, ly + size], radius=4, fill=(9, 34, 60) if target else color)
    ink = (255, 255, 255) if (target or not dashed) else (29, 20, 0)
    d.text((lx + size / 2, ly + size / 2), tag, fill=ink, font=f, anchor="mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma list, e.g. 182,202,208")
    ap.add_argument("--coco_path", default="../data/coco_position.jsonl")
    ap.add_argument("--gen_dir", default="../result/coco_position/")
    ap.add_argument("--det_csv", default="../result/extras_baseline_v3/per_detection.csv")
    ap.add_argument("--out_dir", default="../result/extras_examples/")
    ap.add_argument("--show_wanted", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.coco_path))
    dets = {}
    for r in csv.DictReader(open(args.det_csv)):
        dets.setdefault(r["id"], []).append(r)
    os.makedirs(args.out_dir, exist_ok=True)

    for img_id in args.ids.split(","):
        img = Image.open(os.path.join(args.gen_dir, f"bagel_{img_id}.jpg")).convert("RGB")
        W, H = img.size
        sx, sy = W / DET_SIZE, H / DET_SIZE
        d = ImageDraw.Draw(img)
        lines = [f"bagel_{img_id}: {data[img_id]['caption'].strip()}", ""]

        for t, s in enumerate(data[img_id]["segment"], 1):
            b = [s["bbox"][0] * W, s["bbox"][1] * H, s["bbox"][2] * W, s["bbox"][3] * H]
            draw_box(d, b, f"T{t}", BLUE, W, H, dashed=True, target=True)
            lines.append(f"T{t}\trequested\t{s['label']}")

        e = 0
        for r in dets.get(img_id, []):
            x1, y1, x2, y2 = ast.literal_eval(r["box"])
            b = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]          # exact GroundingDINO box, only rescaled
            if not r["status"].startswith("extra"):
                if args.show_wanted:
                    d.rectangle(b, outline=GREEN, width=2)
                continue
            e += 1
            unsure = bool(r["check_reasons"]) or r.get("low_conf") == "True" or r.get("small") == "True"
            kind = ("check: " + r["check_reasons"] if r["check_reasons"] else
                    "low-confidence" if r.get("low_conf") == "True" else
                    "small" if r.get("small") == "True" else "confirmed")
            draw_box(d, b, f"E{e}", AMBER if unsure else RED, W, H, dashed=unsure)
            lines.append(f"E{e}\t{r['status']}\t{r['type']}\tscore {float(r['score']):.2f}\t{kind}")

        img.save(os.path.join(args.out_dir, f"bagel_{img_id}.jpg"), quality=92)
        with open(os.path.join(args.out_dir, f"bagel_{img_id}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"bagel_{img_id}: {len(data[img_id]['segment'])} requested, {e} extras drawn")


if __name__ == "__main__":
    main()
