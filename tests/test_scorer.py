import pytest

from app.scorer import score

TEXT = "S0 is advice. S1 fact one. S2 fact two. S3 fact three."


def span(sub):
    s = TEXT.index(sub)
    return s, s + len(sub)


def gold_row(aid, sub, label="supported"):
    s, e = span(sub)
    return {"answer_id": aid, "claim_id": f"{aid}:{s}:{e}", "start_char": s, "end_char": e,
            "quote": sub, "expected_label": label, "difficulty_tags": ""}


def pred(aid, sub, label="supported"):
    s, e = span(sub)
    return {"claim_id": f"{aid}:{s}:{e}", "start_char": s, "end_char": e, "quote": sub,
            "label": label, "fact_ids": [], "reason": "r."}


FACTS = ["S1 fact one.", "S2 fact two.", "S3 fact three."]
GOLD = [gold_row("A1", f) for f in FACTS]
TEXTS = {"A1": TEXT, "A2": TEXT}


def ok(aid, claims):
    return {"answer_id": aid, "status": "ok", "claims": claims, "error": None}


def m(result, role):
    return result["metrics"][role]


def test_perfect_output():
    r = score(["A1"], TEXTS, [ok("A1", [pred("A1", f) for f in FACTS])], GOLD)
    assert m(r, "extraction")["f1"] == 1.0 and m(r, "joint")["f1"] == 1.0
    assert r["missed_gold"] == [] and r["unmatched_predictions"] == []


def test_missed_claim():
    r = score(["A1"], TEXTS, [ok("A1", [pred("A1", f) for f in FACTS[:2]])], GOLD)
    assert r["counts"]["M"] == 2
    assert m(r, "extraction") == {"precision": 1.0, "recall": 2 / 3, "f1": 4 / 5}
    assert [x["claim_id"] for x in r["missed_gold"]] == [GOLD[2]["claim_id"]]


def test_extra_advice_prediction_lowers_precision():
    preds = [pred("A1", "S0 is advice.")] + [pred("A1", f) for f in FACTS]
    r = score(["A1"], TEXTS, [ok("A1", preds)], GOLD)
    assert m(r, "extraction")["precision"] == 0.75 and m(r, "extraction")["recall"] == 1.0
    assert r["unmatched_predictions"][0]["quote"] == "S0 is advice."


def test_wrong_label_hits_joint_only():
    preds = [pred("A1", FACTS[0], "unknown")] + [pred("A1", f) for f in FACTS[1:]]
    r = score(["A1"], TEXTS, [ok("A1", preds)], GOLD)
    assert m(r, "extraction")["f1"] == 1.0
    assert m(r, "joint")["f1"] == pytest.approx(2 / 3)
    assert r["wrong_label"][0]["predicted"] == "unknown"


def test_partial_sentence_does_not_match():
    s = TEXT.index("S1 fact")
    partial = {"claim_id": f"A1:{s}:{s + 7}", "start_char": s, "end_char": s + 7, "quote": "S1 fact",
               "label": "supported", "fact_ids": [], "reason": "r."}
    r = score(["A1"], TEXTS, [ok("A1", [partial])], GOLD)
    assert r["counts"]["M"] == 0 and r["counts"]["P"] == 1


def test_provider_error_answer_keeps_gold_as_missed():
    gold = GOLD + [gold_row("A2", f) for f in FACTS]
    outcomes = [ok("A1", [pred("A1", f) for f in FACTS]),
                {"answer_id": "A2", "status": "error", "claims": [],
                 "error": {"type": "provider_error", "message": "x"}}]
    r = score(["A1", "A2"], TEXTS, outcomes, gold)
    assert r["counts"]["G"] == 6 and r["counts"]["error_answers"] == 1
    assert m(r, "joint")["recall"] == 0.5
    assert all(x["answer_error"] for x in r["missed_gold"])


def test_valid_empty_predictions():
    r = score(["A1"], TEXTS, [ok("A1", [])], GOLD)
    assert r["counts"]["empty_answers"] == 1 and r["counts"]["P"] == 0
    assert m(r, "extraction") == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_assessment_arithmetic_example():
    """G=10, P=10, M=8, C=7 -> extraction 0.80, joint 0.70 (not 0.875)."""
    words = [f"Fact number {i}." for i in range(12)]
    text = " ".join(words)
    texts = {"A1": text}

    def row(i, label):
        s = text.index(words[i])
        return s, s + len(words[i]), words[i], label

    gold = [{"answer_id": "A1", "claim_id": f"A1:{s}:{e}", "start_char": s, "end_char": e, "quote": q,
             "expected_label": "supported", "difficulty_tags": ""} for s, e, q, _ in (row(i, "") for i in range(10))]
    preds = []
    for i in list(range(8)) + [10, 11]:          # 8 matched, 2 extra
        s, e, q, _ = row(i, "")
        preds.append({"claim_id": f"A1:{s}:{e}", "start_char": s, "end_char": e, "quote": q,
                      "label": "unknown" if i == 0 else "supported", "fact_ids": [], "reason": "r."})
    r = score(["A1"], texts, [ok("A1", preds)], gold)
    assert (r["counts"]["G"], r["counts"]["P"], r["counts"]["M"], r["counts"]["C"]) == (10, 10, 8, 7)
    for k in ("precision", "recall", "f1"):
        assert m(r, "extraction")[k] == pytest.approx(0.80)
        assert m(r, "joint")[k] == pytest.approx(0.70)


def test_rejects_missing_and_duplicated_outcomes():
    with pytest.raises(ValueError, match="missing"):
        score(["A1", "A2"], TEXTS, [ok("A1", [])], GOLD)
    with pytest.raises(ValueError, match="duplicated"):
        score(["A1"], TEXTS, [ok("A1", []), ok("A1", [])], GOLD)


def test_rejects_prediction_quote_mismatch():
    bad = {**pred("A1", FACTS[0]), "quote": "edited"}
    with pytest.raises(ValueError, match="quote mismatch"):
        score(["A1"], TEXTS, [ok("A1", [bad])], GOLD)


def test_check_split_quality_stays_null():
    r = score(["A1"], TEXTS, [ok("A1", [pred("A1", FACTS[0])])], None)
    assert r["metrics"] is None and r["counts"]["G"] is None and r["counts"]["M"] is None
    assert r["missed_gold"] is None and r["counts"]["P"] == 1


def test_empty_scheduled_split_null_metrics():
    r = score([], TEXTS, [], [])
    assert r["metrics"] is None
