"""CSV loading. Uses the csv module with newline='' so quoted commas and embedded
line breaks stay data; ai_answer is never normalised or re-saved."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANSWER_COLUMNS = ["answer_id", "split", "brand_id", "question", "ai_answer"]
FACT_COLUMNS = ["fact_id", "brand_id", "model_id", "field", "value", "value_type",
                "unit", "condition", "valid_from", "valid_to"]
GOLD_COLUMNS = ["answer_id", "claim_id", "start_char", "end_char", "quote", "expected_label",
                "evidence_fact_ids", "explanation", "difficulty_tags"]


def _read(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != columns:
            raise ValueError(f"{path.name}: unexpected header {reader.fieldnames}")
        return [dict(row) for row in reader]


def parse_value(value: str, value_type: str) -> Any:
    if value_type == "number":
        num = float(value)
        return int(num) if num.is_integer() else num
    if value_type == "boolean":
        if value == "true":
            return True
        if value == "false":
            return False
        raise ValueError(f"invalid boolean text {value!r}")
    if value_type == "string":
        return value
    raise ValueError(f"unknown value_type {value_type!r}")


@dataclass(frozen=True)
class Fact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: Any
    value_type: str
    unit: str | None
    condition: str | None
    valid_from: str
    valid_to: str

    def as_row(self) -> list:
        return [self.fact_id, self.model_id, self.field, self.value, self.unit,
                self.condition, self.valid_from, self.valid_to]


FACT_ROW_COLUMNS = ["fact_id", "model_id", "field", "value", "unit", "condition",
                    "valid_from", "valid_to"]


def load_answers(path: Path) -> list[dict[str, str]]:
    rows = _read(path, ANSWER_COLUMNS)
    ids = [r["answer_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate answer_id in answers.csv")
    return rows


def load_facts(path: Path) -> list[Fact]:
    facts = []
    for r in _read(path, FACT_COLUMNS):
        facts.append(Fact(
            fact_id=r["fact_id"], brand_id=r["brand_id"], model_id=r["model_id"],
            field=r["field"], value=parse_value(r["value"], r["value_type"]),
            value_type=r["value_type"], unit=r["unit"] or None,
            condition=r["condition"] or None, valid_from=r["valid_from"], valid_to=r["valid_to"],
        ))
    if len({f.fact_id for f in facts}) != len(facts):
        raise ValueError("duplicate fact_id in brand_facts.csv")
    return facts


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows = []
    for r in _read(path, GOLD_COLUMNS):
        r = dict(r)
        r["start_char"] = int(r["start_char"])
        r["end_char"] = int(r["end_char"])
        r["evidence_fact_ids"] = [x for x in r["evidence_fact_ids"].split("|") if x]
        rows.append(r)
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "answers": data_dir / "answers.csv",
        "facts": data_dir / "brand_facts.csv",
        "gold": data_dir / "development_gold.csv",
    }
