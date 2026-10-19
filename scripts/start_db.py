"""Start a local embedded PostgreSQL (pgserver) and create the app and test databases.
Prints the DATABASE_URL to put in .env. Any normal PostgreSQL 14+ works instead."""
from __future__ import annotations

from pathlib import Path

import pgserver

ROOT = Path(__file__).resolve().parent.parent

srv = pgserver.get_server(str(ROOT / ".pgdata"), cleanup_mode=None)
for name in ("kasparro", "kasparro_test"):
    if name not in srv.psql(f"SELECT datname FROM pg_database WHERE datname = '{name}';"):
        srv.psql(f"CREATE DATABASE {name};")
base = srv.get_uri().rsplit("/", 1)[0]
print(f"DATABASE_URL={base}/kasparro")
print(f"TEST_DATABASE_URL={base}/kasparro_test")
