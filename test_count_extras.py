"""Hand-computed tests for the counting rule. Run: python -m pytest test_count_extras.py -q
These need no GPU and no GroundingDINO: they feed fake detections into the rule."""
import pytest
from count_extras import iou, ioa, object_type, analyse_type, analyse_image, precision_recall


def test_box_maths():
    big = [300, 40, 500, 290]            # 200 x 250 = 50,000
    b = [320, 60, 440, 160]              # 120 x 100 = 12,000, fully inside big
    assert iou(big, b) == pytest.approx(12000 / 50000)   # 0.24
    assert ioa(b, big) == pytest.approx(1.0)
    assert ioa([0, 0, 10, 10], [5, 0, 20, 10]) == pytest.approx(0.5)
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_object_type():
    assert object_type("a chair") == "chair"
    assert object_type("an apple") == "apple"
    assert object_type("a dining table") == "dining table"
    assert object_type("a pizza") == "pizza"   # official script's replace("a ","") would break words


def test_bagel_202_like_A_B_C():
    """The A/B/C picture: one requested chair box, three detected chairs."""
    red = [300, 40, 500, 290]
    A = {"box": [330, 130, 480, 280], "score": 0.80}   # IoU 22,500/50,000 = 0.45 (< 0.5) -> try 2
    B = {"box": [320, 60, 440, 160], "score": 0.40}    # inside red, partly behind A
    C = {"box": [60, 130, 180, 260], "score": 0.75}    # far away
    res, missing = analyse_type([red], [A, B, C])
    assert missing == 0
    assert [r["status"] for r in res] == ["wanted_try2", "extra_inside", "extra_outside"]
    # B overlaps A: (440-330) * (160-130) = 3,300 of B's 12,000 = 0.275 >= 0.2 -> hidden
    assert res[1]["hidden"] is True
    assert res[2]["hidden"] is False
    # score ratios against the wanted chair's score 0.80
    assert res[1]["score_ratio"] == pytest.approx(0.40 / 0.80)
    assert res[2]["score_ratio"] == pytest.approx(0.75 / 0.80, abs=1e-3)


def test_normal_pair_try1():
    req = [0, 0, 100, 100]
    det = {"box": [5, 5, 100, 100], "score": 0.9}       # IoU 9,025/10,000 = 0.9025
    res, missing = analyse_type([req], [det])
    assert res[0]["status"] == "wanted" and missing == 0


def test_stacked_boxes_bagel_211_like():
    """3 requested boxes almost on top of each other, only 1 object drawn."""
    reqs = [[0, 0, 100, 100], [5, 0, 105, 100], [0, 5, 100, 105]]
    det = {"box": [0, 0, 100, 100], "score": 0.9}
    res, missing = analyse_type(reqs, [det])
    assert [r["status"] for r in res] == ["wanted"]       # paired once, not 3 times
    assert missing == 2


def test_teddy_bagel_208_like():
    """Chair hidden behind the bear on purpose: only the chair back is detected."""
    chair_req = [0, 0, 100, 200]                          # 20,000
    chair_back = {"box": [0, 0, 60, 80], "score": 0.5}    # 4,800 -> IoU 0.24, IoA 1.0
    res, missing = analyse_type([chair_req], [chair_back])
    assert res[0]["status"] == "wanted_try2"
    assert missing == 0


def test_nothing_found():
    res, missing = analyse_type([[0, 0, 10, 10]], [])
    assert res == [] and missing == 1


def test_analyse_image_counts_and_flags():
    segments = [
        {"bbox": [300, 40, 500, 290], "label": "a chair"},
        {"bbox": [0, 300, 200, 500], "label": "a dining table"},
    ]
    dets = {
        "chair": [{"box": [330, 130, 480, 280], "score": 0.8},
                  {"box": [320, 60, 440, 160], "score": 0.4},
                  {"box": [60, 130, 180, 260], "score": 0.75}],
        "dining table": [{"box": [0, 300, 200, 500], "score": 0.9}],
    }
    per_det, s = analyse_image(segments, dets)
    assert s["requested"] == 2 and s["wanted"] == 2 and s["missing"] == 0
    assert s["extra_inside"] == 1 and s["extra_outside"] == 1 and s["extras"] == 2
    assert s["low_score_extras"] == 1          # B: 0.4 / 0.8 = 0.5 < 0.6
    assert set(s["review_flags"]) == {"try2_pairing", "extra_inside", "low_score_extra"}


# ---- check rules (added after the first real run on bagel_208 / bagel_211)

def test_bagel_208_false_chair_is_check_not_extra():
    """Real numbers from the run: GroundingDINO called the bear-on-chair area a 'chair' (0.334)."""
    s512 = lambda b: [v * 512 for v in b]
    segments = [{"bbox": s512([0.097, 0.064, 0.401, 0.655]), "label": "a chair"},
                {"bbox": s512([0.095, 0.103, 0.838, 0.762]), "label": "a teddy bear"}]
    dets = {"chair": [{"box": [51.2, 33.5, 184.1, 335.0], "score": 0.514},     # real chair back
                      {"box": [51.2, 33.6, 427.3, 337.0], "score": 0.334}],    # bear seen as chair
            "teddy bear": [{"box": [51.9, 47.1, 428.7, 388.0], "score": 0.514}]}
    per_det, s = analyse_image(segments, dets)
    assert s["wanted"] == 2 and s["missing"] == 0
    assert s["extras"] == 0                       # the false chair is not counted
    assert s["check"] == 1 and "check" in s["review_flags"]
    false_chair = per_det[1]
    assert "looks_like_teddy_bear" in false_chair["check_reasons"]


def test_low_score_extra_goes_to_lowconf_column():
    """bagel_211: tiny far-away 'motorcycle' with score 0.267 -> low-confidence column, not main count."""
    segments = [{"bbox": [0, 0, 100, 100], "label": "a motorcycle"}]
    dets = {"motorcycle": [{"box": [0, 0, 100, 100], "score": 0.77},
                           {"box": [450, 300, 480, 330], "score": 0.267}]}
    per_det, s = analyse_image(segments, dets)
    assert s["extras"] == 0 and s["check"] == 0 and s["extra_lowconf"] == 1
    assert per_det[1]["low_conf"] is True


def test_real_extra_inside_big_box_of_other_type_is_counted():
    """bagel_175 case: small extra chair inside the big table box is a REAL extra."""
    segments = [{"bbox": [0, 250, 500, 512], "label": "a dining table"},   # big box
                {"bbox": [300, 350, 380, 500], "label": "a chair"}]
    dets = {"dining table": [{"box": [0, 250, 500, 512], "score": 0.8}],
            "chair": [{"box": [300, 350, 380, 500], "score": 0.7},
                      {"box": [100, 380, 170, 500], "score": 0.6}]}       # extra chair at the table
    per_det, s = analyse_image(segments, dets)
    assert s["extra_outside"] == 1 and s["check"] == 0   # counted, not sent to check


def test_inside_extra_scoring_higher_than_wanted_is_check():
    red = [300, 40, 500, 290]
    A = {"box": [305, 45, 495, 285], "score": 0.40}    # paired by IoU, weak score
    B = {"box": [320, 60, 440, 160], "score": 0.80}    # inside, stronger -> pairing may be wrong
    per_det, s = analyse_image([{"bbox": red, "label": "a chair"}], {"chair": [A, B]})
    assert per_det[1]["status"] == "extra_inside"
    assert per_det[1]["check_reasons"] == ["scores_higher_than_wanted"]
    assert s["extras"] == 0 and s["check"] == 1


def test_small_extra_counted_separately():
    """Tiny far-away chair (< 1% of a 512x512 image = 2,621 px^2) goes to its own column."""
    segments = [{"bbox": [0, 0, 200, 200], "label": "a chair"}]
    dets = {"chair": [{"box": [0, 0, 200, 200], "score": 0.8},
                      {"box": [400, 100, 440, 140], "score": 0.6},     # 40x40 = 1,600 px^2 -> small
                      {"box": [300, 300, 400, 450], "score": 0.6}]}    # 100x150 -> normal extra
    per_det, s = analyse_image(segments, dets)
    assert s["extra_small"] == 1 and s["extras"] == 1
    assert per_det[1]["small"] is True and per_det[2]["small"] is False


def test_precision_recall_baseline_numbers():
    """Hand-computed with the baseline totals: 3,200 requested, 2,985 wanted, 608 confirmed, 592 low-conf, 226 small."""
    rows = [{"wanted": 2985, "requested": 3200, "extras": 608, "extra_lowconf": 592, "extra_small": 226}]
    pr = precision_recall(rows)
    assert pr["recall"] == pytest.approx(2985 / 3200, abs=1e-4)                       # 0.9328
    assert pr["precision_confirmed"] == pytest.approx(2985 / (2985 + 608), abs=1e-4)  # 0.8308
    assert pr["precision_all_extras"] == pytest.approx(2985 / (2985 + 1426), abs=1e-4)  # 0.6767


def test_precision_recall_no_objects():
    pr = precision_recall([{"wanted": 0, "requested": 0, "extras": 0, "extra_lowconf": 0, "extra_small": 0}])
    assert pr["recall"] is None and pr["precision_confirmed"] is None


# ---------------------------------------------------------------- v5: cross-label NMS (resolve_double_labels)
from count_extras import resolve_double_labels


def test_v5_bear_also_seen_as_chair_is_removed():
    # bagel_208-like: requested chair (small, chair back) + requested teddy bear (big)
    req = {"chair": [[50, 40, 200, 330]], "teddy bear": [[50, 50, 430, 390]]}
    dets = {"chair": [{"box": [52, 42, 198, 328], "score": 0.45},     # real chair -> protected
                      {"box": [50, 35, 430, 340], "score": 0.60}],    # "chair" on the bear, scores HIGHER
            "teddy bear": [{"box": [55, 50, 425, 385], "score": 0.55}]}
    out, removed = resolve_double_labels(dets, req)
    assert removed == 1
    assert len(out["chair"]) == 1 and out["chair"][0]["score"] == 0.45
    assert len(out["teddy bear"]) == 1
    per_det, summary = analyse_image(
        [{"bbox": b, "label": t} for t, bs in req.items() for b in bs], out)
    assert summary["extras"] == 0 and summary["check"] == 0 and summary["wanted"] == 2


def test_v5_two_requested_objects_that_overlap_both_stay():
    # person on a horse: both requested, boxes overlap a lot -> both protected, nothing removed
    req = {"person": [[100, 50, 300, 400]], "horse": [[80, 120, 320, 420]]}
    dets = {"person": [{"box": [102, 52, 298, 398], "score": 0.7}],
            "horse": [{"box": [82, 122, 318, 418], "score": 0.6}]}
    out, removed = resolve_double_labels(dets, req)
    assert removed == 0 and len(out["person"]) == 1 and len(out["horse"]) == 1


def test_v5_same_label_and_far_boxes_untouched():
    # two real chairs far apart + a table elsewhere: no clash -> nothing removed
    req = {"chair": [[0, 0, 100, 100]], "dining table": [[200, 200, 450, 450]]}
    dets = {"chair": [{"box": [0, 0, 100, 100], "score": 0.8}, {"box": [400, 0, 500, 100], "score": 0.7}],
            "dining table": [{"box": [200, 200, 450, 450], "score": 0.9}]}
    out, removed = resolve_double_labels(dets, req)
    assert removed == 0 and len(out["chair"]) == 2


def test_v5_unrequested_double_label_keeps_higher_score():
    # an extra object seen as both types, neither matches a requested box -> keep the higher score
    req = {"cat": [[0, 0, 100, 100]], "dog": [[400, 400, 500, 500]]}
    dets = {"cat": [{"box": [0, 0, 100, 100], "score": 0.8}, {"box": [200, 200, 300, 300], "score": 0.4}],
            "dog": [{"box": [400, 400, 500, 500], "score": 0.8}, {"box": [202, 202, 300, 300], "score": 0.5}]}
    out, removed = resolve_double_labels(dets, req)
    assert removed == 1
    assert len(out["cat"]) == 1 and len(out["dog"]) == 2   # the 0.5 "dog" wins over the 0.4 "cat"


# ---------------------------------------------------------------- v6: object drawn bigger than its box
def test_v6_skis_bigger_than_box_is_wanted_try3():
    # GenEval bagel_00039: small requested box, one pair of skis drawn over most of the image
    R = [180, 100, 430, 200]
    skis = {"box": [0, 0, 512, 430], "score": 0.62}      # covers all of R, IoU only 0.11
    per_det, s = analyse_image([{"bbox": R, "label": "a pair of skis"}], {"pair of skis": [skis]})
    assert per_det[0]["status"] == "wanted_try3"
    assert s["wanted"] == 1 and s["wanted_try3"] == 1 and s["missing"] == 0 and s["extras"] == 0
    assert "try3_pairing" in s["review_flags"]


def test_v6_airplane_body_and_whole_plane_is_same_object():
    # GenEval bagel_00059: body box pairs (try 1), whole-plane box covers the same requested box
    R = [150, 100, 360, 310]
    body = {"box": [152, 102, 358, 308], "score": 0.90}
    whole = {"box": [0, 100, 512, 310], "score": 0.60}   # IoU with R 0.41 -> no pair
    per_det, s = analyse_image([{"bbox": R, "label": "an airplane"}], {"airplane": [body, whole]})
    assert [r["status"] for r in per_det] == ["wanted", "same_object"]
    assert s["wanted"] == 1 and s["extras"] == 0 and s["same_object"] == 1
    assert "same_object" in s["review_flags"]


def test_v6_real_extra_next_to_wanted_stays_extra():
    # GenEval bagel_00510: second green cow on the grass, below the requested cow box
    R = [300, 50, 450, 245]
    cow = {"box": [302, 52, 448, 243], "score": 0.45}
    extra_cow = {"box": [230, 235, 483, 465], "score": 0.70}  # covers only ~5% of R
    per_det, s = analyse_image([{"bbox": R, "label": "a green cow"}], {"green cow": [cow, extra_cow]})
    assert [r["status"] for r in per_det] == ["wanted", "extra_outside"]
    assert s["extras"] == 1 and s["same_object"] == 0


def test_v6_try2_comes_before_try3():
    # a small detection inside R and a big one around R: the small one pairs (try 2 as in v5)
    R = [100, 100, 200, 200]
    small = {"box": [120, 120, 180, 180], "score": 0.6}   # IoU 0.36, inside R
    big = {"box": [50, 50, 300, 300], "score": 0.7}       # covers all of R, IoU 0.16
    res, missing = analyse_type([R], [small, big])
    assert [r["status"] for r in res] == ["wanted_try2", "same_object"] and missing == 0


def test_v6_off_gives_v5_result():
    R = [180, 100, 430, 200]
    skis = {"box": [0, 0, 512, 430], "score": 0.62}
    per_det, s = analyse_image([{"bbox": R, "label": "a pair of skis"}], {"pair of skis": [skis]}, try3=False)
    assert per_det[0]["status"] == "extra_outside" and s["missing"] == 1 and s["extras"] == 1


# ---------------------------------------------------------------- v7: extras of unrequested types
from count_extras import analyse_unrequested, detect_unrequested, COCO_CLASSES


def _req_scene():
    segments = [{"bbox": [300, 50, 450, 245], "label": "a green cow"},
                {"bbox": [100, 150, 200, 300], "label": "a red umbrella"}]
    dets = {"green cow": [{"box": [302, 52, 448, 243], "score": 0.45},
                          {"box": [230, 235, 483, 465], "score": 0.70}],        # extra cow (requested type)
            "red umbrella": [{"box": [100, 150, 200, 300], "score": 0.9}]}
    per_det, s = analyse_image(segments, dets)
    return segments, per_det, s


def test_v7_alias_of_requested_object_not_counted():
    segments, per_det, _ = _req_scene()
    unreq = {"cow": [{"box": [301, 51, 449, 244], "score": 0.8},      # the wanted cow, seen as "cow"
                     {"box": [232, 236, 480, 466], "score": 0.7}],    # the requested-type extra cow, again
             "umbrella": [{"box": [101, 151, 199, 299], "score": 0.9}]}
    rows, c = analyse_unrequested(unreq, segments, per_det)
    assert c["unreq_alias"] == 3 and c["unreq_extras"] == 0


def test_v7_unrequested_outside_inside_lowconf_small():
    segments, per_det, _ = _req_scene()
    unreq = {"person": [{"box": [0, 330, 150, 500], "score": 0.6}],         # outside all boxes
             "bird": [{"box": [110, 160, 180, 230], "score": 0.5},          # inside the umbrella box (IoU < 0.5)
                      {"box": [10, 10, 40, 40], "score": 0.6}],             # 900 px^2 < 1% -> small
             "dog": [{"box": [20, 60, 120, 140], "score": 0.3}]}            # score < 0.35 -> lowconf
    rows, c = analyse_unrequested(unreq, segments, per_det)
    st = {(r["type"], tuple(r["box"])): r["status"] for r in rows}
    assert st[("person", (0, 330, 150, 500))] == "unreq_outside"
    assert st[("bird", (110, 160, 180, 230))] == "unreq_inside"
    assert st[("bird", (10, 10, 40, 40))] == "unreq_small"
    assert st[("dog", (20, 60, 120, 140))] == "unreq_lowconf"
    assert c["unreq_extras"] == 2 and c["unreq_outside"] == 1 and c["unreq_inside"] == 1


def test_v7_precision_incl_unrequested():
    rows = [{"wanted": 90, "requested": 100, "extras": 10, "extra_lowconf": 0, "extra_small": 0, "unreq_extras": 20}]
    pr = precision_recall(rows)
    assert pr["precision_confirmed"] == pytest.approx(0.9)
    assert pr["precision_incl_unrequested"] == pytest.approx(90 / 120, abs=1e-4)
    assert precision_recall([{k: v for k, v in rows[0].items() if k != "unreq_extras"}])["precision_incl_unrequested"] is None


def test_v7_detect_unrequested_chunks_words(monkeypatch):
    import count_extras as ce
    calls = []

    def fake_joint(model, image_np, words):
        calls.append(list(words))
        return {w: ([{"box": [0, 0, 10, 10], "score": 0.5}] if w == "person" else []) for w in words}, 0
    monkeypatch.setattr(ce, "detect_joint", fake_joint)
    words = [w for w in COCO_CLASSES if w != "chair"]                       # 79 words
    out = detect_unrequested(None, None, words)
    assert [len(c) for c in calls] == [20, 20, 20, 19]
    assert len(out) == 79 and len(out["person"]) == 1 and "chair" not in out


def test_main_metric_precision_outside():
    """Main metric (Ilyas): extras of requested types outside the boxes. Inside extras do not lower it."""
    rows = [{"wanted": 90, "requested": 100, "extras": 15, "extra_outside": 10, "extra_inside": 5,
             "extra_lowconf": 0, "extra_small": 0}]
    pr = precision_recall(rows)
    assert pr["precision_outside"] == pytest.approx(90 / 100)
    assert pr["precision_confirmed"] == pytest.approx(90 / 105, abs=1e-4)
