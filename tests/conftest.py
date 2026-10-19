from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.data import data_paths, load_answers, load_facts, load_gold  # noqa: E402
from app.provider import ChatResult, ToolCall  # noqa: E402
from app.tools import FactStore  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    return replace(get_settings(), reports_dir=tmp_path_factory.mktemp("reports"))


@pytest.fixture(scope="session")
def paths(settings):
    return data_paths(settings.data_dir)


@pytest.fixture(scope="session")
def store(paths):
    return FactStore(load_facts(paths["facts"]))


@pytest.fixture(scope="session")
def answers(paths):
    return {a["answer_id"]: a for a in load_answers(paths["answers"])}


@pytest.fixture(scope="session")
def gold(paths):
    return load_gold(paths["gold"])


def tool_call(brand="cleartrail", name="get_brand_facts", call_id="c1"):
    return ChatResult(None, [ToolCall(call_id, name, json.dumps({"brand_id": brand}))],
                      finish_reason="tool_calls")


def final(claims, answer_id=None):
    body = {"claims": claims}
    if answer_id:
        body["answer_id"] = answer_id
    return ChatResult(json.dumps(body), finish_reason="stop")


def agent_input_from(messages):
    return json.loads(messages[1]["content"].split("\n", 1)[1])


def first_sentences(text, n=3, skip=3):
    return [m.group(0).strip() for m in re.finditer(r"[^.!?\n]+[.!?]", text)][skip:skip + n]


@pytest.fixture(scope="session")
def test_db_url():
    url = os.getenv("TEST_DATABASE_URL")
    if not url and os.getenv("DATABASE_URL"):
        url = os.environ["DATABASE_URL"].rsplit("/", 1)[0] + "/kasparro_test"
    if not url:
        pytest.skip("no TEST_DATABASE_URL / DATABASE_URL configured")
    try:
        import psycopg
        psycopg.connect(url, connect_timeout=3).close()
    except Exception as exc:
        pytest.skip(f"test database unreachable: {exc}")
    return url


@pytest.fixture()
def db(test_db_url):
    from app.db import Database

    database = Database(test_db_url)
    database.init_schema()
    with database.connect() as conn:
        conn.execute("TRUNCATE claim_results, answer_results, runs")
    return database
