"""PostgreSQL persistence (psycopg 3, parameterized SQL only)."""
from __future__ import annotations

import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    split TEXT NOT NULL CHECK (split IN ('development', 'check')),
    agent_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    fact_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    settings JSONB NOT NULL,
    source_revision TEXT,
    file_hashes JSONB NOT NULL,
    counts JSONB,
    metrics JSONB,
    details JSONB,
    failure TEXT
);
CREATE TABLE IF NOT EXISTS answer_results (
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    answer_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error_type TEXT,
    error_message TEXT,
    latency_ms INTEGER,
    claim_count INTEGER NOT NULL DEFAULT 0,
    trace_ref TEXT,
    PRIMARY KEY (run_id, answer_id),
    CHECK ((status = 'error') = (error_type IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS claim_results (
    run_id UUID NOT NULL,
    answer_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char INTEGER NOT NULL,
    quote TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('supported', 'contradicted', 'unknown')),
    fact_ids JSONB NOT NULL,
    reason TEXT NOT NULL,
    matched_gold_id TEXT,
    expected_label TEXT,
    label_correct BOOLEAN,
    PRIMARY KEY (run_id, answer_id, start_char, end_char),
    UNIQUE (run_id, claim_id),
    CHECK (end_char > start_char),
    FOREIGN KEY (run_id, answer_id) REFERENCES answer_results(run_id, answer_id) ON DELETE CASCADE
);
"""


class Database:
    def __init__(self, url: str):
        self.url = url

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.url, row_factory=dict_row)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(SCHEMA)

    def create_run(self, run_id: uuid.UUID, meta: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO runs (run_id, status, split, agent_version, prompt_version,
                       dataset_version, evaluator_version, fact_version, provider, model,
                       settings, source_revision, file_hashes)
                   VALUES (%s, 'running', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (run_id, meta["split"], meta["agent_version"], meta["prompt_version"],
                 meta["dataset_version"], meta["evaluator_version"], meta["fact_version"],
                 meta["provider"], meta["model"], Jsonb(meta["settings"]),
                 meta.get("source_revision"), Jsonb(meta["file_hashes"])))

    def save_answer(self, run_id: uuid.UUID, outcome: dict[str, Any]) -> None:
        err = outcome.get("error") or {}
        with self.connect() as conn, conn.transaction():
            conn.execute(
                """INSERT INTO answer_results (run_id, answer_id, status, error_type, error_message,
                       latency_ms, claim_count, trace_ref)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (run_id, outcome["answer_id"], outcome["status"], err.get("type"), err.get("message"),
                 outcome.get("latency_ms"), len(outcome.get("claims", [])), outcome.get("trace_ref")))
            for c in outcome.get("claims", []):
                conn.execute(
                    """INSERT INTO claim_results (run_id, answer_id, claim_id, start_char, end_char,
                           quote, label, fact_ids, reason)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (run_id, outcome["answer_id"], c["claim_id"], c["start_char"], c["end_char"],
                     c["quote"], c["label"], Jsonb(c["fact_ids"]), c["reason"]))

    def save_matches(self, run_id: uuid.UUID, matches: dict[str, dict]) -> None:
        with self.connect() as conn, conn.transaction():
            for claim_id, m in matches.items():
                conn.execute(
                    """UPDATE claim_results SET matched_gold_id = %s, expected_label = %s,
                           label_correct = %s WHERE run_id = %s AND claim_id = %s""",
                    (m["gold_claim_id"], m["expected_label"], m["label_correct"], run_id, claim_id))

    def finish_run(self, run_id: uuid.UUID, status: str, counts: dict | None, metrics: dict | None,
                   details: dict | None, failure: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE runs SET status = %s, finished_at = now(), counts = %s, metrics = %s,
                       details = %s, failure = %s WHERE run_id = %s""",
                (status, Jsonb(counts), Jsonb(metrics), Jsonb(details), failure, run_id))

    def get_run(self, run_id: uuid.UUID) -> dict[str, Any] | None:
        with self.connect() as conn:
            run = conn.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,)).fetchone()
            if run is None:
                return None
            rows = conn.execute(
                """SELECT a.answer_id, a.status, a.error_type, a.error_message, a.latency_ms,
                          a.claim_count, a.trace_ref,
                          c.claim_id, c.start_char, c.end_char, c.quote, c.label, c.fact_ids,
                          c.reason, c.matched_gold_id, c.expected_label, c.label_correct
                   FROM answer_results a
                   LEFT JOIN claim_results c
                     ON c.run_id = a.run_id AND c.answer_id = a.answer_id
                   WHERE a.run_id = %s
                   ORDER BY a.answer_id, c.start_char, c.end_char""",
                (run_id,)).fetchall()
        answers: dict[str, dict] = {}
        for r in rows:
            a = answers.setdefault(r["answer_id"], {
                "answer_id": r["answer_id"], "status": r["status"],
                "error": ({"type": r["error_type"], "message": r["error_message"]}
                          if r["error_type"] else None),
                "latency_ms": r["latency_ms"], "trace_ref": r["trace_ref"], "claims": [],
            })
            if r["claim_id"] is not None:
                a["claims"].append({k: r[k] for k in (
                    "claim_id", "start_char", "end_char", "quote", "label", "fact_ids", "reason",
                    "matched_gold_id", "expected_label", "label_correct")})
        run = dict(run)
        run["run_id"] = str(run["run_id"])
        for k in ("created_at", "finished_at"):
            run[k] = run[k].isoformat() if run[k] else None
        run["answers"] = list(answers.values())
        return run
