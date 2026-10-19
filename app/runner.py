"""Evaluation runner: sequential, one terminal outcome per scheduled answer.
Predictions are saved before gold is joined (gold never touches the inference path)."""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import AgentInput, AgentRunError, prompt_version, run_agent
from .config import DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION, ROOT, Settings
from .data import data_paths, load_answers, load_facts, load_gold, sha256_file
from .db import Database
from .provider import Provider
from .scorer import score
from .tools import FactStore

SPLITS = ("development", "check")


def source_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:
        return None


def code_hash() -> str:
    """Hash of agent code + prompts, so a run is tied to the exact source even without git."""
    import hashlib

    h = hashlib.sha256()
    for p in sorted((ROOT / "app").rglob("*")):
        if p.suffix in (".py", ".md"):
            h.update(p.relative_to(ROOT).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def classify_one(inp: AgentInput, version: str, provider: Provider, store: FactStore,
                 settings: Settings) -> dict[str, Any]:
    """Run the agent on one answer and return a terminal outcome record."""
    t0 = time.perf_counter()
    try:
        claims, trace = run_agent(inp, version, provider, store, settings.max_turns)
        outcome = {"answer_id": inp.answer_id, "status": "ok", "claims": claims, "error": None}
    except AgentRunError as err:
        trace = err.trace
        outcome = {"answer_id": inp.answer_id, "status": "error", "claims": [], "error": err.as_dict()}
    except Exception as exc:  # unexpected agent bug: still a preserved answer-level error
        trace = {}
        outcome = {"answer_id": inp.answer_id, "status": "error", "claims": [],
                   "error": {"type": "internal_error", "message": type(exc).__name__}}
    outcome["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    outcome["trace"] = trace
    return outcome


def run_evaluation(agent_version: str, split: str, provider: Provider, settings: Settings,
                   db: Database | None = None, write_report: bool = True,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    paths = data_paths(settings.data_dir)
    answers = [a for a in load_answers(paths["answers"]) if a["split"] == split]
    store = FactStore(load_facts(paths["facts"]))
    run_id = uuid.uuid4()
    started = datetime.now(timezone.utc)
    meta = {
        "run_id": str(run_id), "split": split, "agent_version": agent_version,
        "prompt_version": prompt_version(agent_version), "dataset_version": DATASET_VERSION,
        "evaluator_version": EVALUATOR_VERSION, "fact_version": FACT_VERSION,
        "provider": provider.name, "model": provider.model, "settings": settings.public(),
        "source_revision": source_revision() or f"code-sha256:{code_hash()}",
        "started_at": started.isoformat(),
        "file_hashes": {name: sha256_file(p) for name, p in paths.items()},
    }
    if db:
        db.create_run(run_id, meta)
    trace_dir = settings.reports_dir / "traces" / str(run_id)
    outcomes: list[dict[str, Any]] = []
    status, failure = "completed", None
    try:
        for i, a in enumerate(answers, 1):
            inp = AgentInput(a["answer_id"], a["brand_id"], a["question"], a["ai_answer"])
            outcome = classify_one(inp, agent_version, provider, store, settings)
            if write_report:
                trace_dir.mkdir(parents=True, exist_ok=True)
                ref = trace_dir / f"{a['answer_id']}.json"
                ref.write_text(json.dumps(outcome.pop("trace"), indent=1, ensure_ascii=False), encoding="utf-8")
                outcome["trace_ref"] = (ref.relative_to(ROOT) if ref.is_relative_to(ROOT) else ref).as_posix()
            else:
                outcome.pop("trace")
            outcomes.append(outcome)
            if db:
                db.save_answer(run_id, outcome)
            if progress:
                err = outcome["error"]["type"] if outcome["error"] else ""
                progress(f"[{i}/{len(answers)}] {a['answer_id']} {outcome['status']} "
                         f"claims={len(outcome['claims'])} {err} {outcome['latency_ms']}ms")
            if outcome["error"] and outcome["error"]["type"] == "provider_rate_limited" \
                    and "per day" in outcome["error"]["message"]:
                raise RuntimeError("provider daily quota exhausted; run interrupted")
    except (Exception, KeyboardInterrupt) as exc:
        status, failure = "failed", f"{type(exc).__name__}: {exc}"[:300]

    texts = {a["answer_id"]: a["ai_answer"] for a in answers}
    scheduled = [a["answer_id"] for a in answers]
    result = None
    if status == "completed":
        gold = load_gold(paths["gold"]) if split == "development" else None
        result = score(scheduled, texts, outcomes, gold)
    report = {
        "run": {**meta, "status": status, "failure": failure,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
                "token_usage": _usage(settings, trace_dir) if write_report else None},
        "counts": result["counts"] if result else {"processed_answers": len(outcomes),
                                                   "scheduled_answers": len(scheduled)},
        "metrics": result["metrics"] if result else None,
        "error_answer_ids": result["error_answer_ids"] if result else None,
        "missed_gold": result["missed_gold"] if result else None,
        "unmatched_predictions": result["unmatched_predictions"] if result else None,
        "wrong_label": result["wrong_label"] if result else None,
        "answers": [{**o, "claims": [{**c, **(result["matches"].get(c["claim_id"], {}) if result else {})}
                                     for c in o["claims"]]} for o in outcomes],
    }
    if db:
        if result:
            db.save_matches(run_id, result["matches"])
        details = {k: report[k] for k in ("error_answer_ids", "missed_gold", "unmatched_predictions",
                                          "wrong_label")}
        details["missed_gold_ids"] = [m["claim_id"] for m in report["missed_gold"] or []]
        details["unmatched_prediction_ids"] = [m["claim_id"] for m in report["unmatched_predictions"] or []]
        details["wrong_label_ids"] = [m["claim_id"] for m in report["wrong_label"] or []]
        db.finish_run(run_id, status, report["counts"], report["metrics"], details, failure)
    if write_report:
        write_reports(report, settings.reports_dir)
    return report


def _usage(settings: Settings, trace_dir: Path) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "model_turns": 0}
    for f in trace_dir.glob("*.json"):
        for t in json.loads(f.read_text(encoding="utf-8")).get("turns", []):
            total["model_turns"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                total[k] += t.get("usage", {}).get(k, 0) or 0
    return total


def write_reports(report: dict[str, Any], reports_dir: Path) -> Path:
    run = report["run"]
    name = f"{run['agent_version']}_{run['split']}_{run['started_at'][:19].replace(':', '')}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{name}.json"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (reports_dir / f"{name}.md").write_text(render_markdown(report), encoding="utf-8")
    return json_path


def _fmt(x: Any) -> str:
    return "null" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def render_markdown(report: dict[str, Any]) -> str:
    run, counts, m = report["run"], report["counts"], report["metrics"]
    lines = [f"# Run {run['agent_version']} / {run['split']}", "",
             f"- run_id: `{run['run_id']}`  status: **{run['status']}**" +
             (f"  failure: {run['failure']}" if run["failure"] else ""),
             f"- provider/model: {run['provider']} / {run['model']}  prompt: {run['prompt_version']}",
             f"- dataset {run['dataset_version']}, facts {run['fact_version']}, evaluator {run['evaluator_version']}",
             f"- settings: `{json.dumps(run['settings'])}`",
             f"- started {run['started_at']}  elapsed {run['elapsed_s']}s  tokens {run.get('token_usage')}",
             "", "## Counts", "", "| " + " | ".join(counts) + " |", "|" + "---|" * len(counts),
             "| " + " | ".join(_fmt(v) for v in counts.values()) + " |", ""]
    lines += ["## Metrics", ""]
    if m is None:
        lines.append("Quality metrics are null (unlabeled check split or incomplete run).")
    else:
        lines += ["| role | precision | recall | F1 |", "|---|---|---|---|"]
        for role in ("extraction", "joint"):
            lines.append(f"| {role} | {_fmt(m[role]['precision'])} | {_fmt(m[role]['recall'])} | "
                         f"{_fmt(m[role]['f1'])} |")
        lines.append(f"\nGold claim coverage C/G: {_fmt(m['gold_claim_coverage'])}")
    lines += ["", "## Answer errors", ""]
    errs = [a for a in report["answers"] if a["status"] == "error"]
    lines += [f"- {a['answer_id']}: {a['error']['type']} - {a['error']['message']}" for a in errs] or ["None."]
    for key, title in (("missed_gold", "Missed gold claims"), ("unmatched_predictions", "Extra predictions"),
                       ("wrong_label", "Matched with wrong label")):
        rows = report.get(key)
        if rows is None:
            continue
        lines += ["", f"## {title} ({len(rows)})", ""]
        for r in rows:
            label = r.get("expected") or r.get("label")
            extra = f" predicted={r['predicted']}" if "predicted" in r else ""
            lines.append(f"- `{r['claim_id']}` {label}{extra}: {r['quote'][:140]}")
    return "\n".join(lines) + "\n"
