"""Read-only fact lookup tool and its constrained executor."""
from __future__ import annotations

import json
from typing import Any

from .config import FACT_VERSION
from .data import FACT_ROW_COLUMNS, Fact
from .errors import AgentError

TOOL_NAME = "get_brand_facts"


class FactStore:
    def __init__(self, facts: list[Fact]):
        self.facts = facts
        self.by_id = {f.fact_id: f for f in facts}
        self.by_brand: dict[str, list[Fact]] = {}
        for f in facts:
            self.by_brand.setdefault(f.brand_id, []).append(f)

    @property
    def brands(self) -> list[str]:
        return sorted(self.by_brand)

    def get_brand_facts(self, brand_id: str) -> dict[str, Any]:
        if brand_id not in self.by_brand:
            raise AgentError("unknown_brand", f"unknown brand_id {brand_id!r}")
        return {
            "brand_id": brand_id,
            "fact_version": FACT_VERSION,
            "columns": FACT_ROW_COLUMNS,
            "rows": [f.as_row() for f in self.by_brand[brand_id]],
        }


def tool_spec(store: FactStore) -> dict:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": ("Return every catalogue fact row for one brand (fact_version "
                            f"{FACT_VERSION}). Known brand_ids: {', '.join(store.brands)}."),
            "parameters": {
                "type": "object",
                "properties": {"brand_id": {"type": "string", "description": "Catalogue brand id"}},
                "required": ["brand_id"],
                "additionalProperties": False,
            },
        },
    }


def execute_tool_call(store: FactStore, name: str, arguments: str) -> dict[str, Any]:
    """Validate the requested tool and its arguments, then run the lookup.
    Only get_brand_facts exists; nothing from answer text is ever executed."""
    if name != TOOL_NAME:
        raise AgentError("illegal_tool", f"model requested unknown tool {name!r}")
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        raise AgentError("invalid_tool_arguments", f"tool arguments are not JSON: {exc.msg}") from None
    if not isinstance(args, dict) or set(args) != {"brand_id"} or not isinstance(args["brand_id"], str):
        raise AgentError("invalid_tool_arguments", "tool arguments must be exactly {brand_id: string}")
    return store.get_brand_facts(args["brand_id"])
