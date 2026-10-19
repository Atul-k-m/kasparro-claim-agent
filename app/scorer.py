"""Deterministic scorer: exact (answer_id, start_char, end_char) matching, extraction and joint
precision / recall / F1. No fuzzy matching and no LLM judge."""
from __future__ import annotations

from typing import Any

from .config import LABELS


def _prf(x: int, p: int, g: int) -> dict[str, float]:
    return {
        "precision": x / p if p else 0.0,
        "recall": x / g if g else 0.0,
        "f1": 2 * x / (p + g) if (p + g) else 0.0,
    }


def check_outcomes(scheduled_ids: list[str], outcomes: list[dict[str, Any]]) -> None:
    """Exactly one terminal outcome per scheduled answer."""
    ids = [o["answer_id"] for o in outcomes]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicated answer outcomes: {dupes}")
    missing = sorted(set(scheduled_ids) - set(ids))
    if missing:
        raise ValueError(f"missing answer outcomes: {missing}")
    extra = sorted(set(ids) - set(scheduled_ids))
    if extra:
        raise ValueError(f"outcomes for unscheduled answers: {extra}")
    for o in outcomes:
        if o["status"] not in ("ok", "error"):
            raise ValueError(f"bad status for {o['answer_id']}: {o['status']}")
        if o["status"] == "error" and o.get("claims"):
            raise ValueError(f"error answer {o['answer_id']} cannot carry accepted predictions")


def score(scheduled_ids: list[str], answer_texts: dict[str, str], outcomes: list[dict[str, Any]],
          gold: list[dict[str, Any]] | None) -> dict[str, Any]:
    """gold=None means an unlabeled (check) split: quality fields stay null."""
    check_outcomes(scheduled_ids, outcomes)
    preds = [(o["answer_id"], c) for o in outcomes if o["status"] == "ok" for c in o.get("claims", [])]
    for aid, c in preds:
        if c["quote"] != answer_texts[aid][c["start_char"]:c["end_char"]]:
            raise ValueError(f"prediction quote mismatch for {c['claim_id']}")
        if c["label"] not in LABELS:
            raise ValueError(f"prediction label invalid for {c['claim_id']}")
    errors = sorted(o["answer_id"] for o in outcomes if o["status"] == "error")
    counts = {
        "scheduled_answers": len(scheduled_ids),
        "ok_answers": len(scheduled_ids) - len(errors),
        "error_answers": len(errors),
        "empty_answers": sum(1 for o in outcomes if o["status"] == "ok" and not o.get("claims")),
        "P": len(preds), "G": None, "M": None, "C": None,
    }
    result: dict[str, Any] = {
        "counts": counts, "metrics": None, "error_answer_ids": errors,
        "missed_gold": None, "unmatched_predictions": None, "wrong_label": None, "matches": {},
    }
    if gold is None:
        return result

    scheduled = set(scheduled_ids)
    gold = [g for g in gold if g["answer_id"] in scheduled]
    for g in gold:
        if g["quote"] != answer_texts[g["answer_id"]][g["start_char"]:g["end_char"]]:
            raise ValueError(f"gold quote mismatch for {g['claim_id']}")
    by_span = {(g["answer_id"], g["start_char"], g["end_char"]): g for g in gold}
    used: set[str] = set()
    matches: dict[str, dict] = {}
    unmatched, wrong = [], []
    for aid, c in preds:
        g = by_span.get((aid, c["start_char"], c["end_char"]))
        if g is None or g["claim_id"] in used:
            unmatched.append({"claim_id": c["claim_id"], "label": c["label"], "quote": c["quote"]})
            continue
        used.add(g["claim_id"])
        correct = c["label"] == g["expected_label"]
        matches[c["claim_id"]] = {"gold_claim_id": g["claim_id"], "expected_label": g["expected_label"],
                                  "label_correct": correct}
        if not correct:
            wrong.append({"claim_id": g["claim_id"], "predicted": c["label"],
                          "expected": g["expected_label"], "quote": g["quote"],
                          "difficulty_tags": g["difficulty_tags"]})
    missed = [{"claim_id": g["claim_id"], "expected": g["expected_label"], "quote": g["quote"],
               "answer_error": g["answer_id"] in errors}
              for g in gold if g["claim_id"] not in used]
    G, P, M = len(gold), len(preds), len(matches)
    C = sum(1 for m in matches.values() if m["label_correct"])
    counts.update(G=G, M=M, C=C)
    result.update(
        matches=matches, missed_gold=missed, unmatched_predictions=unmatched, wrong_label=wrong,
        metrics=None if G == 0 else {
            "extraction": _prf(M, P, G),
            "joint": _prf(C, P, G),
            "gold_claim_coverage": C / G,
        },
    )
    return result
