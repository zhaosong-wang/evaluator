"""
Sort the drawn result images into one folder per category (as links, so no extra disk space).

An image goes into EVERY folder that applies, e.g. an image with an outside extra and a missing
object is in both "outside_extra" and "missing".

Folders (inside <result_dir>/sorted/):
  clean          no extras of any kind, nothing missing, nothing to check
  outside_extra  >= 1 confirmed extra outside the boxes
  inside_extra   >= 1 confirmed extra inside a box
  lowconf_extra  >= 1 low-confidence extra (score 0.25-0.35)
  small_extra    >= 1 small extra (< 1% of the image)
  check          >= 1 extra marked "check" (looks like another requested type / pairing doubt)
  missing        >= 1 requested object not found

Run from eval/MIG_Bench, after count_extras.py drew the images (without --no_draw):
  python sort_by_category.py --result_dir ../result/extras_baseline_v3/
"""
import argparse
import csv
import os

RULES = {
    "outside_extra": lambda r: int(r["extra_outside"]) > 0,
    "inside_extra": lambda r: int(r["extra_inside"]) > 0,
    "lowconf_extra": lambda r: int(r["extra_lowconf"]) > 0,
    "small_extra": lambda r: int(r["extra_small"]) > 0,
    "check": lambda r: int(r["check"]) > 0,
    "missing": lambda r: int(r["missing"]) > 0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.result_dir, "per_image.csv"))))
    vis = os.path.join(args.result_dir, "vis")
    sorted_dir = os.path.join(args.result_dir, "sorted")
    counts = {name: 0 for name in list(RULES) + ["clean"]}

    for name in counts:
        os.makedirs(os.path.join(sorted_dir, name), exist_ok=True)

    for r in rows:
        img = f"bagel_{r['id']}.jpg"
        if not os.path.exists(os.path.join(vis, img)):
            print(f"no drawn image for {r['id']} (run count_extras.py without --no_draw)")
            continue
        hits = [name for name, rule in RULES.items() if rule(r)] or ["clean"]
        for name in hits:
            link = os.path.join(sorted_dir, name, img)
            if not os.path.lexists(link):
                os.symlink(os.path.join("..", "..", "vis", img), link)   # relative link
            counts[name] += 1

    print(f"{len(rows)} images sorted into {sorted_dir}/")
    for name, n in counts.items():
        print(f"  {name:14s} {n:4d}")


if __name__ == "__main__":
    main()
