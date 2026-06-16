"""Vroomarr — FastAPI entry point.

Environment variables:
    TSDB_API_KEY    TheSportsDB API key (default: "1" — free public key)
    TSDB_PREMIUM    Set to "1" if you have a premium TSDB key
    DEFAULT_RACE_HOURS  Fallback race duration in hours (default: 3.0)
    PORT            Server port (default: 9198)
    LOG_LEVEL       Logging level (default: INFO)
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import date as Date

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from motorsports.service import MotorsportService
from motorsports.types import MatchResult, RacingEvent, SessionWindow

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

_service: MotorsportService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = MotorsportService(
        tsdb_api_key=os.environ.get("TSDB_API_KEY"),
        tsdb_premium=bool(os.environ.get("TSDB_PREMIUM")),
        default_race_hours=float(os.environ.get("DEFAULT_RACE_HOURS", "3.0")),
    )
    yield
    if _service:
        _service.close()


app = FastAPI(
    title="Vroomarr",
    description=(
        "Match IPTV/EPG stream names to motorsports events and compute per-session time windows. "
        "Supports F1, NASCAR, IndyCar, MotoGP (ESPN) and WEC/IMSA (TSDB + static calendar)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SessionOut(BaseModel):
    code: str
    name: str
    start: str  # ISO 8601
    end: str


class EventOut(BaseModel):
    id: str
    provider: str
    name: str
    short_name: str
    start_time: str
    league: str
    circuit_name: str | None
    sessions: list[dict]  # RacingSession list (code/name/start_time)


class MatchRequest(BaseModel):
    stream_name: str
    date: Date
    league: str


class MatchResponse(BaseModel):
    matched: bool
    method: str | None = None
    confidence: float = 0.0
    event: EventOut | None = None
    sessions: list[SessionOut] = []
    reason: str | None = None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _session_out(w: SessionWindow) -> SessionOut:
    return SessionOut(
        code=w.code,
        name=w.name,
        start=w.start.isoformat(),
        end=w.end.isoformat(),
    )


def _event_out(e: RacingEvent) -> EventOut:
    return EventOut(
        id=e.id,
        provider=e.provider,
        name=e.name,
        short_name=e.short_name,
        start_time=e.start_time.isoformat(),
        league=e.league,
        circuit_name=e.circuit_name,
        sessions=[
            {"code": s.code, "name": s.name, "start_time": s.start_time.isoformat()}
            for s in e.sessions
        ],
    )


def _match_response(result: MatchResult) -> MatchResponse:
    return MatchResponse(
        matched=result.matched,
        method=result.method,
        confidence=result.confidence,
        event=_event_out(result.event) if result.event else None,
        sessions=[_session_out(s) for s in result.sessions],
        reason=result.reason,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/leagues")
def get_leagues():
    """List all supported racing leagues."""
    return _service.get_leagues()


@app.get("/events/{league}", response_model=list[EventOut])
def get_events(
    league: str,
    date: Date | None = Query(default=None, description="Target date (YYYY-MM-DD). Defaults to today."),
):
    """Return racing events for a league on the given date."""
    target = date or Date.today()
    events = _service.get_events(league, target)
    return [_event_out(e) for e in events]


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    """Match a stream name to a racing event and return per-session time windows.

    Example request:
        {"stream_name": "F1: Monaco Grand Prix", "date": "2026-06-01", "league": "f1"}
    """
    if req.league not in {lg["code"] for lg in _service.get_leagues()}:
        raise HTTPException(status_code=404, detail=f"Unknown league: {req.league!r}")

    result = _service.match(req.stream_name, req.league, req.date)
    return _match_response(result)


@app.post("/match/batch", response_model=list[MatchResponse])
def match_batch(requests: list[MatchRequest]):
    """Match multiple streams in one call."""
    return [_match_response(_service.match(r.stream_name, r.league, r.date)) for r in requests]
