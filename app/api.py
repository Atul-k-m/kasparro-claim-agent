"""Thin FastAPI layer over the agent, runner and database."""
from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any, Callable, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentInput, AgentRunError, run_agent
from .config import Settings, get_settings
from .data import data_paths, load_facts
from .db import Database
from .errors import HTTP_STATUS
from .provider import Provider, build_provider
from .runner import run_evaluation
from .tools import FactStore


class ClassifyRequest(BaseModel):
    # extra="forbid": any gold / claim-list / label field in the request is a 422.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    answer_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    brand_id: str = Field(min_length=1, max_length=64)
    question: str = Field(max_length=5000)
    ai_answer: str = Field(min_length=1, max_length=20000)
    agent_version: Literal["v1", "v2"]


class Claim(BaseModel):
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: Literal["supported", "contradicted", "unknown"]
    fact_ids: list[str]
    reason: str


class ClassifyResponse(BaseModel):
    answer_id: str
    agent_version: str
    claims: list[Claim]


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_version: Literal["v1", "v2"]
    split: Literal["development", "check"]


def _error(status: int, type_: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": type_, "message": message}})


def create_app(settings: Settings | None = None,
               provider_factory: Callable[[], Provider] | None = None,
               db: Database | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = FactStore(load_facts(data_paths(settings.data_dir)["facts"]))
    if db is None and settings.database_url:
        db = Database(settings.database_url)
    if db is not None:
        db.init_schema()
    make_provider = provider_factory or (lambda: build_provider(settings))
    app = FastAPI(title="Kasparro claim classifier", version="1.0")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Report field locations only; never echo request values back.
        fields = sorted({".".join(str(p) for p in e["loc"] if p != "body") for e in exc.errors()})
        return _error(422, "invalid_request", f"invalid or forbidden fields: {', '.join(fields)}")

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest) -> Any:
        if req.brand_id not in store.brands:
            return _error(404, "unknown_brand", "primary brand_id is not in the catalogue")
        inp = AgentInput(req.answer_id, req.brand_id, req.question, req.ai_answer)
        try:
            provider = make_provider()
            claims, _ = run_agent(inp, req.agent_version, provider, store, settings.max_turns)
        except AgentRunError as err:
            return _error(HTTP_STATUS.get(err.type, 502), err.type, err.message)
        except Exception as exc:  # provider construction or unexpected failure
            return _error(502, "provider_error", type(exc).__name__)
        return {"answer_id": req.answer_id, "agent_version": req.agent_version, "claims": claims}

    @app.post("/evaluation-runs", status_code=201)
    def create_run(req: EvaluationRunRequest) -> Any:
        if db is None:
            return _error(503, "database_unavailable", "DATABASE_URL is not configured")
        run_settings = replace(settings, reports_dir=settings.reports_dir / "api_runs")
        report = run_evaluation(req.agent_version, req.split, make_provider(), run_settings, db)
        return {
            "run_id": report["run"]["run_id"], "status": report["run"]["status"],
            "failure": report["run"]["failure"], "split": req.split,
            "agent_version": req.agent_version, "counts": report["counts"],
            "metrics": report["metrics"], "error_answer_ids": report["error_answer_ids"],
        }

    @app.get("/evaluation-runs/{run_id}")
    def get_run(run_id: uuid.UUID) -> Any:
        if db is None:
            return _error(503, "database_unavailable", "DATABASE_URL is not configured")
        run = db.get_run(run_id)
        if run is None:
            return _error(404, "run_not_found", "no evaluation run with that id")
        return run

    return app


def app_factory() -> FastAPI:
    return create_app()
