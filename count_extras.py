"""
Count unwanted (extra) objects in generated images.

Rule (designed by Ilyas, see project doc "baseline-extra-objects-observations"):
  Per image, per requested object type (e.g. "chair"):
  1. Detect   GroundingDINO finds all boxes of that type (score >= BOX_THRESHOLD), then NMS.
  2. Pair     Each requested box gets at most one detection, and each detection at most one
              requested box (one-to-one, highest IoU first).
              Try 1: IoU >= 0.5.
              Try 2: for requested boxes still unpaired, a detection that lies mostly inside
                     the box (IoA >= 0.5), highest IoU first.  -> status "wanted_try2"
  3. Extras   Every detection that is not paired.
  4. Where    IoA of the extra with the requested boxes of the same type:
              >= 0.5 -> "extra_inside", else "extra_outside".
  5. Hidden   An extra that overlaps another detection of the same type gets hidden=True.

Check rules (not counted as extras, put on the review list instead):
  - IoU >= CROSS_TYPE_IOU with a requested box of another type (likely that object, bagel_208)
  - inside a box and scoring higher than the object paired to that box (pairing may be wrong)

Score check (Ilyas's idea): every detection keeps its GroundingDINO score. An extra whose score
is much lower than the wanted objects of the same type (score_ratio < LOW_SCORE_RATIO) is flagged,
so these images can be checked by eye.

Run from inside eval/MIG_Bench (needs the GroundingDINO config + checkpoint there):
  python count_extras.py --coco_path ../data/coco_position.jsonl \
      --gen_dir ../result/coco_position/ --out_dir ../result/extras_baseline/
"""

import argparse
import csv
import json
import os
import re

import numpy as np

# ---------------------------------------------------------------- settings (all are choices, not rules)
BOX_THRESHOLD = 0.25     # same as the official COCO-Position script
TEXT_THRESHOLD = 0.25    # same as the official COCO-Position script
NMS_IOU = 0.5            # official script uses 0.8; 0.5 removes more duplicate boxes of one object
PAIR_IOU = 0.5           # try 1: same success threshold as the official script
PAIR_IOA = 0.5           # try 2 and inside/outside split
HIDDEN_IOA = 0.2         # extra counts as "hidden" if >= 20% of it overlaps another detection
LOW_SCORE_RATIO = 0.6    # extra score < 60% of the median wanted score of its type -> flag
EXTRA_MIN_SCORE = 0.35   # extras below this score go to their own "low-confidence" column
                         # (manual check of 30: ~83% real, mostly tiny/far, partly rendered or group boxes)
CROSS_TYPE_IOU = 0.5     # extra almost the same box as a requested object of another type -> "check" (bagel_208)
SMALL_AREA_FRAC = 0.01   # extra smaller than 1% of the image (~51x51 px at 512) -> counted separately as "small"
IMG_SIZE = 512           # official script resizes images to 512x512 before detection


# ---------------------------------------------------------------- box maths (boxes are [x1, y1, x2, y2])
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(inner, outer):
    """Share of `inner`'s own area that lies inside `outer`."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def object_type(label):
    """'a chair' -> 'chair', 'an apple' -> 'apple' (only the leading article)."""
    return re.sub(r"^(a|an|the)\s+", "", label.strip().lower())


# ---------------------------------------------------------------- the rule (pure function, no GPU needed)
def analyse_type(requested, detections):
    """
    requested:  list of boxes asked for (one object type)
    detections: list of dicts {"box": [...], "score": float} for the same type
    Returns (det_results, n_missing) where det_results is one dict per detection.
    """
    n_req, n_det = len(requested), len(detections)
    pair_of_det = [None] * n_det      # index of the requested box, or None
    how = [None] * n_det              # "try1" / "try2"
    req_taken = [False] * n_req

    # Try 1: one-to-one, highest IoU first, IoU >= PAIR_IOU
    cands = []
    for r, rb in enumerate(requested):
        for d, det in enumerate(detections):
            v = iou(rb, det["box"])
            if v >= PAIR_IOU:
                cands.append((v, r, d))
    for v, r, d in sorted(cands, reverse=True):
        if not req_taken[r] and pair_of_det[d] is None:
            req_taken[r], pair_of_det[d], how[d] = True, r, "try1"

    # Try 2: still-unpaired requested boxes; detection mostly inside the box (IoA >= PAIR_IOA)
    cands = []
    for r, rb in enumerate(requested):
        if req_taken[r]:
            continue
        for d, det in enumerate(detections):
            if pair_of_det[d] is None and ioa(det["box"], rb) >= PAIR_IOA:
                cands.append((iou(rb, det["box"]), r, d))
    for v, r, d in sorted(cands, reverse=True):
        if not req_taken[r] and pair_of_det[d] is None:
            req_taken[r], pair_of_det[d], how[d] = True, r, "try2"

    wanted_scores = [detections[d]["score"] for d in range(n_det) if pair_of_det[d] is not None]
    ref_score = float(np.median(wanted_scores)) if wanted_scores else None

    results = []
    for d, det in enumerate(detections):
        ioas = [ioa(det["box"], rb) for rb in requested]
        best_ioa = max(ioas, default=0.0)
        res = {
            "box": [round(float(x), 1) for x in det["box"]],
            "score": round(float(det["score"]), 3),
            "best_iou": round(max((iou(det["box"], rb) for rb in requested), default=0.0), 3),
            "best_ioa": round(best_ioa, 3),
            "hidden": False,
            "score_ratio": None,
            "check_reasons": [],
            "low_conf": False,
        }
        if pair_of_det[d] is not None:
            res["status"] = "wanted" if how[d] == "try1" else "wanted_try2"
        else:
            res["status"] = "extra_inside" if best_ioa >= PAIR_IOA else "extra_outside"
            if res["status"] == "extra_inside":
                # rule 3: inside a box, but scores higher than the object paired to that box
                r_in = ioas.index(best_ioa)
                paired = [k for k in range(n_det) if pair_of_det[k] == r_in]
                if paired and det["score"] > detections[paired[0]]["score"]:
                    res["check_reasons"].append("scores_higher_than_wanted")
            res["low_conf"] = det["score"] < EXTRA_MIN_SCORE
            others = [o["box"] for k, o in enumerate(detections) if k != d]
            res["hidden"] = any(ioa(det["box"], ob) >= HIDDEN_IOA for ob in others)
            if ref_score:
                res["score_ratio"] = round(det["score"] / ref_score, 3)
        results.append(res)

    n_missing = req_taken.count(False)
    return results, n_missing


def analyse_image(segments, detections_by_type, image_area=IMG_SIZE * IMG_SIZE):
    """segments: [{"bbox": [x1,y1,x2,y2] in pixels, "label": "a chair"}, ...]"""
    requested_by_type = {}
    for s in segments:
        requested_by_type.setdefault(object_type(s["label"]), []).append(s["bbox"])

    per_det, summary = [], {"requested": 0, "wanted": 0, "wanted_try2": 0, "missing": 0,
                            "extra_outside": 0, "extra_inside": 0, "extra_hidden": 0, "check": 0, "extra_small": 0, "extra_lowconf": 0,
                            "low_score_extras": 0, "stacked_requested_boxes": 0}
    for t, req in requested_by_type.items():
        res, n_missing = analyse_type(req, detections_by_type.get(t, []))
        summary["requested"] += len(req)
        summary["missing"] += n_missing
        for r in res:
            r["type"] = t
            if r["status"].startswith("extra"):
                # rule 2: almost the same box as a requested object of another type
                for t2, req2 in requested_by_type.items():
                    if t2 != t and any(iou(r["box"], rb) >= CROSS_TYPE_IOU for rb in req2):
                        r["check_reasons"].append(f"looks_like_{t2.replace(' ', '_')}")
            r["small"] = r["status"].startswith("extra") and area(r["box"]) < SMALL_AREA_FRAC * image_area
            if r["status"].startswith("extra") and r["check_reasons"]:
                summary["check"] += 1          # not counted as an extra until checked by eye
                continue
            if r.get("low_conf"):
                summary["extra_lowconf"] += 1  # low detector confidence: own column
                continue
            if r["small"]:
                summary["extra_small"] += 1    # tiny: own column
                continue
            summary[r["status"]] += 1
            if r["status"] == "wanted_try2":
                summary["wanted"] += 1
            if r["status"].startswith("extra"):
                summary["extra_hidden"] += r["hidden"]
                if r["score_ratio"] is not None and r["score_ratio"] < LOW_SCORE_RATIO:
                    summary["low_score_extras"] += 1
        per_det += res
        # requested boxes of the same type that overlap a lot (bagel_211 case)
        for i in range(len(req)):
            for j in range(i + 1, len(req)):
                if iou(req[i], req[j]) >= 0.3:
                    summary["stacked_requested_boxes"] += 1

    summary["extras"] = summary["extra_outside"] + summary["extra_inside"]
    # review flags: things a human should look at
    flags = []
    if summary["wanted_try2"]:
        flags.append("try2_pairing")
    if summary["missing"]:
        flags.append("missing")
    if summary["extra_inside"]:
        flags.append("extra_inside")
    if summary["low_score_extras"]:
        flags.append("low_score_extra")
    if summary["stacked_requested_boxes"]:
        flags.append("stacked_boxes")
    if summary["check"]:
        flags.append("check")
    summary["review_flags"] = flags
    return per_det, summary


def precision_recall(rows):
    """
    Precision and recall over many images (Yujin's suggestion).
      recall    = wanted / requested                 (how many requested objects showed up)
      precision = wanted / (wanted + extras)         (how much of what was drawn was requested)
    Two precisions: confirmed extras only, and all extras (+ low-confidence + small).
    "check" extras are left out of both (unclear whether they are extras at all).
    """
    wanted = sum(r["wanted"] for r in rows)
    requested = sum(r["requested"] for r in rows)
    confirmed = sum(r["extras"] for r in rows)
    everything = confirmed + sum(r["extra_lowconf"] + r["extra_small"] for r in rows)
    div = lambda a, b: round(a / b, 4) if b else None
    return {"recall": div(wanted, requested),
            "precision_confirmed": div(wanted, wanted + confirmed),
            "precision_all_extras": div(wanted, wanted + everything),
            "wanted_objects": wanted}


# ---------------------------------------------------------------- detection + drawing (needs GPU + GroundingDINO)
def nms(dets, thr):
    import torch
    import torchvision
    if len(dets) < 2:
        return dets
    boxes = torch.tensor([d["box"] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, thr).tolist()
    return [dets[k] for k in keep]


def detect(model, image_np, name):
    det = model.predict_with_classes(image=image_np, classes=[name],
                                     box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
    dets = [{"box": b.tolist(), "score": float(s)} for b, s in zip(det.xyxy, det.confidence)]
    return nms(dets, NMS_IOU)


COLORS = {"requested": (220, 40, 40), "wanted": (30, 160, 80), "wanted_try2": (30, 160, 80),
          "extra_outside": (240, 160, 20), "extra_inside": (150, 60, 200), "check": (120, 120, 120),
          "small": (60, 140, 230), "lowconf": (230, 210, 40)}


def draw(image_pil, segments, per_det, path):
    from PIL import ImageDraw
    img = image_pil.copy()
    d = ImageDraw.Draw(img)
    for s in segments:
        d.rectangle(s["bbox"], outline=COLORS["requested"], width=2)
    for r in per_det:
        style = (COLORS["check"] if r["check_reasons"] else
                 COLORS["lowconf"] if r.get("low_conf") else
                 COLORS["small"] if r.get("small") else COLORS[r["status"]])
        d.rectangle(r["box"], outline=style, width=3 if r["status"].startswith("extra") else 2)
        tag = (f'{r["type"]} {r["status"].replace("extra_", "x-")} {r["score"]:.2f}' + (" H" if r["hidden"] else "")
               + (" CHECK" if r["check_reasons"] else " LOW" if r.get("low_conf") else
                  " SMALL" if r.get("small") else ""))
        d.text((r["box"][0] + 3, r["box"][1] + 3), tag, fill=style)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_path", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ids", default=None, help="comma list, e.g. 182,202,208,211 (for testing)")
    ap.add_argument("--no_draw", action="store_true")
    ap.add_argument("--from_csv", default=None,
                    help="per_detection.csv of an earlier run: re-apply the rules to its detections, no GPU")
    args = ap.parse_args()

    from PIL import Image
    saved = None
    if args.from_csv:
        import ast
        saved = {}
        for r in csv.DictReader(open(args.from_csv)):
            saved.setdefault(r["id"], {}).setdefault(r["type"], []).append(
                {"box": ast.literal_eval(r["box"]), "score": float(r["score"])})
        model = None
    else:
        from groundingdino.util.inference import Model
        model = Model(model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                      model_checkpoint_path="./groundingdino_swint_ogc.pth")

    data = json.load(open(args.coco_path))
    ids = args.ids.split(",") if args.ids else list(data.keys())
    os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)

    det_rows, img_rows = [], []
    for n, img_id in enumerate(ids):
        path = os.path.join(args.gen_dir, f"bagel_{img_id}.jpg")
        if not os.path.exists(path):
            print(f"skip {img_id}: no image")
            continue
        image_pil = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))  # same as official script
        image_np = np.array(image_pil)
        segments = [{"bbox": [v * IMG_SIZE for v in s["bbox"]], "label": s["label"]}
                    for s in data[img_id]["segment"]]
        types = sorted({object_type(s["label"]) for s in segments})
        if saved is not None:
            dets = {t: saved.get(img_id, {}).get(t, []) for t in types}
        else:
            dets = {t: detect(model, image_np, t) for t in types}
        per_det, summary = analyse_image(segments, dets)

        for r in per_det:
            det_rows.append({"id": img_id, **r, "check_reasons": "|".join(r["check_reasons"])})
        img_rows.append({"id": img_id, "caption": data[img_id]["caption"].strip(), **summary,
                         "review_flags": "|".join(summary["review_flags"])})
        if not args.no_draw:
            draw(image_pil, segments, per_det, os.path.join(args.out_dir, "vis", f"bagel_{img_id}.jpg"))
        print(f"[{n+1}/{len(ids)}] {img_id}: wanted {summary['wanted']}/{summary['requested']}, "
              f"extra out {summary['extra_outside']}, in {summary['extra_inside']}, "
              f"missing {summary['missing']}  {summary['review_flags']}")

    # ---- write results
    with open(os.path.join(args.out_dir, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(img_rows[0].keys()))
        w.writeheader(); w.writerows(img_rows)
    with open(os.path.join(args.out_dir, "per_detection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
        w.writeheader(); w.writerows(det_rows)
    review = sorted([r for r in img_rows if r["review_flags"]], key=lambda r: -len(r["review_flags"].split("|")))
    with open(os.path.join(args.out_dir, "review_list.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(img_rows[0].keys()))
        w.writeheader(); w.writerows(review)

    n = len(img_rows)
    tot = lambda k: sum(r[k] for r in img_rows)
    summary = {
        "images": n,
        "requested_objects": tot("requested"),
        "missing_objects": tot("missing"),
        **precision_recall(img_rows),
        "extras_per_image": round(tot("extras") / n, 3),
        "extra_outside_per_image": round(tot("extra_outside") / n, 3),
        "extra_inside_per_image": round(tot("extra_inside") / n, 3),
        "lowconf_extras_per_image": round(tot("extra_lowconf") / n, 3),
        "images_with_lowconf_extra_pct": round(100 * sum(r["extra_lowconf"] > 0 for r in img_rows) / n, 2),
        "small_extras_per_image": round(tot("extra_small") / n, 3),
        "images_with_small_extra_pct": round(100 * sum(r["extra_small"] > 0 for r in img_rows) / n, 2),
        "images_with_extra_pct": round(100 * sum(r["extras"] > 0 for r in img_rows) / n, 2),
        "extras_to_check": tot("check"),
        "images_to_review": len(review),
        "settings": {"box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD, "nms_iou": NMS_IOU,
                     "pair_iou": PAIR_IOU, "pair_ioa": PAIR_IOA, "hidden_ioa": HIDDEN_IOA,
                     "low_score_ratio": LOW_SCORE_RATIO, "extra_min_score": EXTRA_MIN_SCORE,
                     "cross_type_iou": CROSS_TYPE_IOU, "small_area_frac": SMALL_AREA_FRAC, "img_size": IMG_SIZE},
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
