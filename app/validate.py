"""Parse model output, locate quotes in the exact ai_answer, and validate final claims.
Any failure raises AgentError('invalid_output') so the whole answer becomes an error."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .config import LABELS
from .errors import AgentError

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def _invalid(msg: str) -> AgentError:
    return AgentError("invalid_output", msg)


def parse_model_json(content: str | None) -> dict[str, Any]:
    if not content or not content.strip():
        raise _invalid("empty final response")
    text = _FENCE.sub("", content.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise _invalid("final response is not a JSON object")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise _invalid(f"malformed JSON: {exc.msg}") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        raise _invalid("JSON must be an object with a 'claims' list")
    return obj


def limit_sentences(text: str, n: int = 2) -> str:
    parts = _SENTENCE_END.split(text.strip())
    return " ".join(parts[:n])


def locate_claims(answer_id: str, ai_answer: str, raw: dict[str, Any],
                  known_fact_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Turn model claims (quote + label + fact_ids + reason) into span-anchored claims."""
    if raw.get("answer_id") not in (None, answer_id):
        raise _invalid("answer_id in output does not match input")
    known = set(known_fact_ids)
    taken: list[tuple[int, int]] = []
    cursor = 0
    claims = []
    for i, item in enumerate(raw["claims"]):
        if not isinstance(item, dict):
            raise _invalid(f"claim {i} is not an object")
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            raise _invalid(f"claim {i} has no quote")
        quote = quote.strip()
        start = _find_occurrence(ai_answer, quote, cursor, taken)
        if start is None:
            raise _invalid(f"claim {i} quote is not an exact substring of ai_answer")
        end = start + len(quote)
        label = item.get("label")
        if label not in LABELS:
            raise _invalid(f"claim {i} has unsupported label {label!r}")
        fact_ids = item.get("fact_ids", [])
        if not isinstance(fact_ids, list) or not all(isinstance(f, str) for f in fact_ids):
            raise _invalid(f"claim {i} fact_ids must be a list of strings")
        unknown_ids = [f for f in fact_ids if f not in known]
        if unknown_ids:
            raise _invalid(f"claim {i} cites non-catalogue fact ids {unknown_ids[:3]}")
        reason = item.get("reason", "")
        if not isinstance(reason, str):
            raise _invalid(f"claim {i} reason must be a string")
        claims.append({
            "claim_id": f"{answer_id}:{start}:{end}", "start_char": start, "end_char": end,
            "quote": ai_answer[start:end], "label": label,
            "fact_ids": list(dict.fromkeys(fact_ids)), "reason": limit_sentences(reason),
        })
        taken.append((start, end))
        cursor = end
    claims.sort(key=lambda c: c["start_char"])
    validate_claims(answer_id, ai_answer, claims, known)
    return claims


def _find_occurrence(text: str, quote: str, cursor: int, taken: list[tuple[int, int]]) -> int | None:
    """Prefer the first occurrence after the previous claim (text order), then any free one.
    A repeated sentence is thus resolved by its position among the model's claims."""
    for begin in (cursor, 0):
        pos = text.find(quote, begin)
        while pos >= 0:
            end = pos + len(quote)
            if not any(pos < e and s < end for s, e in taken):
                return pos
            pos = text.find(quote, pos + 1)
    return None


def validate_claims(answer_id: str, ai_answer: str, claims: list[dict[str, Any]],
                    known_fact_ids: Iterable[str]) -> None:
    known = set(known_fact_ids)
    seen: set[str] = set()
    prev_end = -1
    for c in claims:
        s, e, q = c.get("start_char"), c.get("end_char"), c.get("quote")
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(ai_answer)):
            raise _invalid(f"span out of bounds: {s}..{e}")
        if q != ai_answer[s:e]:
            raise _invalid(f"quote does not equal ai_answer[{s}:{e}]")
        if q != q.strip():
            raise _invalid("quote has leading or trailing whitespace")
        if c.get("claim_id") != f"{answer_id}:{s}:{e}":
            raise _invalid("claim_id must be answer_id:start_char:end_char")
        if c["claim_id"] in seen:
            raise _invalid(f"duplicate claim {c['claim_id']}")
        seen.add(c["claim_id"])
        if s < prev_end:
            raise _invalid(f"overlapping or unordered span at {s}")
        prev_end = e
        if c.get("label") not in LABELS:
            raise _invalid(f"unsupported label {c.get('label')!r}")
        if not isinstance(c.get("fact_ids"), list) or any(f not in known for f in c["fact_ids"]):
            raise _invalid("fact_ids must reference existing catalogue facts")
        if not isinstance(c.get("reason"), str) or len(_SENTENCE_END.split(c["reason"].strip())) > 2:
            raise _invalid("reason must be a string of at most two sentences")
