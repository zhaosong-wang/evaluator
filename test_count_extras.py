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
