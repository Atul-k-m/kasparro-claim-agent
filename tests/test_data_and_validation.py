import pytest

from app.data import ANSWER_COLUMNS, load_answers, parse_value
from app.errors import AgentError
from app.validate import limit_sentences, locate_claims, parse_model_json, validate_claims

TEXT = "Intro advice here. Mini costs INR 1,299, tax included.\nMini weighs 320 g. Mini weighs 320 g."


def test_csv_quoted_commas_and_embedded_newlines(tmp_path):
    raw = ('answer_id,split,brand_id,question,ai_answer\r\n'
           'X1,development,cleartrail,"Q, with comma","Line one, with comma.\nLine two ""quoted"".  "\r\n')
    p = tmp_path / "answers.csv"
    p.write_bytes(raw.encode("utf-8"))
    rows = load_answers(p)
    assert list(rows[0]) == ANSWER_COLUMNS
    assert rows[0]["question"] == "Q, with comma"
    assert rows[0]["ai_answer"] == 'Line one, with comma.\nLine two "quoted".  '  # whitespace kept


def test_real_answers_preserved_exactly(answers, gold):
    assert len(answers) == 50 and sum("\n" in a["ai_answer"] for a in answers.values()) == 50
    assert len(gold) == 376
    for g in gold:  # every gold offset slices the decoded text exactly -> no normalisation
        assert answers[g["answer_id"]]["ai_answer"][g["start_char"]:g["end_char"]] == g["quote"]


def test_boolean_and_number_parsing():
    assert parse_value("false", "boolean") is False
    assert parse_value("true", "boolean") is True
    assert parse_value("1299", "number") == 1299
    assert parse_value("5.3", "string") == "5.3"
    with pytest.raises(ValueError):
        parse_value("False", "boolean")


def _claim(quote, label="supported", fact_ids=(), reason="r."):
    return {"quote": quote, "label": label, "fact_ids": list(fact_ids), "reason": reason}


KNOWN = {"cleartrail.mini.price_2026", "cleartrail.mini.weight"}


def test_locate_valid_offsets_and_ids():
    q = "Mini costs INR 1,299, tax included."
    claims = locate_claims("A1", TEXT, {"claims": [_claim(q, fact_ids=["cleartrail.mini.price_2026"])]}, KNOWN)
    c = claims[0]
    assert TEXT[c["start_char"]:c["end_char"]] == q == c["quote"]
    assert c["claim_id"] == f"A1:{c['start_char']}:{c['end_char']}"


def test_repeated_quote_resolved_by_order():
    q = "Mini weighs 320 g."
    claims = locate_claims("A1", TEXT, {"claims": [_claim(q), _claim(q)]}, KNOWN)
    assert [c["start_char"] for c in claims] == [TEXT.index(q), TEXT.rindex(q)]


@pytest.mark.parametrize("raw, msg", [
    ({"claims": [_claim("Mini costs INR 999.")]}, "not an exact substring"),
    ({"claims": [_claim("Mini weighs 320 g.", label="true")]}, "unsupported label"),
    ({"claims": [_claim("Mini weighs 320 g.", fact_ids=["cleartrail.mini.ghost"])]}, "non-catalogue"),
    ({"claims": [_claim("Mini weighs 320 g."), _claim("Mini weighs 320 g."), _claim("Mini weighs 320 g.")]},
     "not an exact substring"),
    ({"answer_id": "A9", "claims": []}, "answer_id"),
])
def test_locate_rejects_invalid(raw, msg):
    with pytest.raises(AgentError) as exc:
        locate_claims("A1", TEXT, raw, KNOWN)
    assert exc.value.type == "invalid_output" and msg in exc.value.message


def _span(s, e, label="unknown"):
    return {"claim_id": f"A1:{s}:{e}", "start_char": s, "end_char": e, "quote": TEXT[s:e],
            "label": label, "fact_ids": [], "reason": "r."}


def test_validate_offsets_quote_duplicates_overlaps():
    s = TEXT.index("Mini costs")
    e = TEXT.index("included.") + len("included.")
    validate_claims("A1", TEXT, [_span(s, e)], KNOWN)  # valid
    bad = [
        [_span(s, len(TEXT) + 5)],                                   # out of bounds
        [{**_span(s, e), "quote": "Mini costs less."}],              # quote inequality
        [_span(s - 1, e)],                                           # leading whitespace
        [_span(s, e), _span(s, e)],                                  # duplicate id
        [_span(s, e), _span(s + 5, e + 3)],                          # overlap
        [{**_span(s, e), "claim_id": "A1:0:1"}],                     # wrong derived id
        [{**_span(s, e), "reason": "One. Two. Three."}],             # reason > 2 sentences
    ]
    for claims in bad:
        with pytest.raises(AgentError):
            validate_claims("A1", TEXT, claims, KNOWN)


def test_parse_model_json_variants():
    assert parse_model_json('```json\n{"claims": []}\n```') == {"claims": []}
    for bad in ["", "no json", '{"claims": {}}', '{"claims": [}']:
        with pytest.raises(AgentError):
            parse_model_json(bad)


def test_reason_limited_to_two_sentences():
    assert limit_sentences("One. Two. Three.") == "One. Two."
