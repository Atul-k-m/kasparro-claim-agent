"""CLI: python scripts/run_eval.py --version v1 --split development"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.provider import build_provider  # noqa: E402
from app.runner import run_evaluation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], required=True)
    ap.add_argument("--split", choices=["development", "check"], required=True)
    ap.add_argument("--no-db", action="store_true", help="skip PostgreSQL persistence")
    args = ap.parse_args()
    settings = get_settings()
    db = None
    if not args.no_db:
        if not settings.database_url:
            print("DATABASE_URL is not set (use --no-db to skip persistence)", file=sys.stderr)
            return 2
        db = Database(settings.database_url)
        db.init_schema()
    report = run_evaluation(args.version, args.split, build_provider(settings), settings, db,
                            progress=lambda m: print(m, flush=True))
    print("status:", report["run"]["status"], report["run"]["failure"] or "")
    print("run_id:", report["run"]["run_id"])
    print("counts:", report["counts"])
    print("metrics:", report["metrics"])
    return 0 if report["run"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
