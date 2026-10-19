import json
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.errors import AgentError
from app.provider import FakeProvider
from conftest import agent_input_from, final, first_sentences, tool_call


def scripted(messages, tools, choice):
    """Fake model: A03 -> provider error, A05 -> valid empty, others -> three sentences."""
    inp = agent_input_from(messages)
    if choice == "required":
        return tool_call(inp["brand_id"])
    if inp["answer_id"] == "A03":
        raise AgentError("provider_error", "HTTP 500: upstream")
    if inp["answer_id"] == "A05":
        return final([])
    return final([{"quote": q, "label": "unknown", "fact_ids": [], "reason": "r."}
                  for q in first_sentences(inp["ai_answer"])])


@pytest.fixture()
def client(settings, db):
    return TestClient(create_app(settings, provider_factory=lambda: FakeProvider(scripted), db=db))


def body(answers, aid="A01", **extra):
    a = answers[aid]
    return {"answer_id": aid, "brand_id": a["brand_id"], "question": a["question"],
            "ai_answer": a["ai_answer"], "agent_version": "v1", **extra}


def test_classify_success(client, answers):
    r = client.post("/classify", json=body(answers))
    assert r.status_code == 200
    claims = r.json()["claims"]
    text = answers["A01"]["ai_answer"]
    assert len(claims) == 3 and all(text[c["start_char"]:c["end_char"]] == c["quote"] for c in claims)


@pytest.mark.parametrize("extra", [{"claims": []}, {"gold": []}, {"expected_label": "supported"},
                                   {"agent_version": "v3"}])
def test_classify_rejects_gold_claims_and_bad_fields(client, answers, extra):
    r = client.post("/classify", json=body(answers, **extra))
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_request"


def test_classify_missing_field_422(client, answers):
    b = body(answers)
    del b["ai_answer"]
    assert client.post("/classify", json=b).status_code == 422


def test_classify_unknown_primary_brand_404(client, answers):
    assert client.post("/classify", json=body(answers, brand_id="acme")).status_code == 404


@pytest.mark.parametrize("err, status", [(AgentError("provider_error", "HTTP 500"), 502),
                                         (AgentError("provider_timeout", "slow"), 504)])
def test_classify_provider_failures(settings, db, answers, err, status):
    app = create_app(settings, provider_factory=lambda: FakeProvider([err]), db=db)
    r = TestClient(app).post("/classify", json=body(answers))
    assert r.status_code == status and set(r.json()["error"]) == {"type", "message"}


def test_classify_invalid_model_output_502(settings, db, answers):
    from app.provider import ChatResult
    app = create_app(settings, provider_factory=lambda: FakeProvider([tool_call(), ChatResult("nope")]), db=db)
    r = TestClient(app).post("/classify", json=body(answers))
    assert r.status_code == 502 and r.json()["error"]["type"] == "invalid_output"


def test_development_run_persists_and_joins(client, db):
    r = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "development"})
    assert r.status_code == 201
    out = r.json()
    assert out["status"] == "completed" and out["counts"]["scheduled_answers"] == 40
    assert out["counts"]["error_answers"] == 1 and out["error_answer_ids"] == ["A03"]
    assert out["counts"]["G"] == 376 and out["metrics"]["joint"]["f1"] is not None

    got = client.get(f"/evaluation-runs/{out['run_id']}").json()
    ids = [a["answer_id"] for a in got["answers"]]
    assert ids == sorted(ids) and len(ids) == 40
    by = {a["answer_id"]: a for a in got["answers"]}
    assert by["A03"]["status"] == "error" and by["A03"]["error"]["type"] == "provider_error"
    assert by["A03"]["claims"] == []
    assert by["A05"]["status"] == "ok" and by["A05"]["claims"] == []
    starts = [c["start_char"] for c in by["A01"]["claims"]]
    assert starts == sorted(starts) and len(starts) == 3
    assert got["metrics"] == out["metrics"] and got["file_hashes"]["answers"]
    matched = [c for a in got["answers"] for c in a["claims"] if c["matched_gold_id"]]
    assert matched and all(c["expected_label"] for c in matched)


def test_check_run_keeps_quality_null(client):
    out = client.post("/evaluation-runs", json={"agent_version": "v2", "split": "check"}).json()
    assert out["metrics"] is None and out["counts"]["G"] is None and out["counts"]["scheduled_answers"] == 10
    got = client.get(f"/evaluation-runs/{out['run_id']}").json()
    assert all(c["matched_gold_id"] is None for a in got["answers"] for c in a["claims"])


def test_run_validation_and_missing(client):
    assert client.post("/evaluation-runs", json={"agent_version": "v1", "split": "final"}).status_code == 422
    assert client.get("/evaluation-runs/not-a-uuid").status_code == 422
    assert client.get(f"/evaluation-runs/{uuid.uuid4()}").status_code == 404


def test_database_constraints(db):
    run_id = uuid.uuid4()
    meta = {"split": "check", "agent_version": "v1", "prompt_version": "v1-x", "dataset_version": "d",
            "evaluator_version": "e", "fact_version": "f", "provider": "fake", "model": "m",
            "settings": {}, "file_hashes": {}}
    db.create_run(run_id, meta)
    db.save_answer(run_id, {"answer_id": "A41", "status": "ok", "claims": []})
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.save_answer(run_id, {"answer_id": "A41", "status": "ok", "claims": []})
    claim = {"claim_id": "A42:0:5", "start_char": 0, "end_char": 5, "quote": "Hello", "label": "unknown",
             "fact_ids": [], "reason": "r."}
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.save_answer(run_id, {"answer_id": "A42", "status": "ok", "claims": [claim, claim]})
    # the failed transaction left no partial answer row behind
    assert [a["answer_id"] for a in db.get_run(run_id)["answers"]] == ["A41"]
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        db.save_answer(uuid.uuid4(), {"answer_id": "A41", "status": "ok", "claims": []})
