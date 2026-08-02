"""
watchdog.py — external watchdog for the KalshiML dashboard.

Lives on the BRIDGE, deliberately. A watchdog inside KalshiML cannot tell you
KalshiML is dead. This is a separate Railway service with its own process, so
it keeps reporting when the thing it watches stops.

Shape
-----
A background asyncio task samples two dashboard endpoints every
WATCHDOG_INTERVAL_SEC, derives a Sample, runs every check against it, and
emails via Resend **only on state transitions** — ok -> bad, and bad -> ok.
A watchdog that speaks every cycle gets muted inside a week, so silence is the
normal output.

Safety properties, in order of how badly they'd bite:

  * OFF by default (WATCHDOG_ENABLED). Two services build from this repo; only
    one should be watching, or every alert arrives twice.
  * Runs as its own task, NOT inside /health. That endpoint is Railway's
    liveness probe — multi-second HTTP work there is how you turn a slow
    dependency into a restart loop.
  * Every cycle is wrapped: an exception marks the cycle failed and sleeps.
    The watchdog must never be the thing that takes the bridge down.
  * Outbound email is rate-capped (MAX_EMAILS_PER_HOUR). A flapping check
    cannot turn into hundreds of messages.
  * No secrets in state or /watchdog/status output.

Known limits (documented rather than hidden):
  * Trend state is IN-MEMORY. A redeploy resets the rolling window, so
    degradation checks go quiet for a few hours after every deploy. Persisting
    it needs a volume the bridge does not currently have.
  * Sampling reads the dashboard's live log buffer, which only holds lines
    since the KalshiML process started. Right after a KalshiML restart the
    window is short and rate-based checks stay unarmed until it fills.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _b(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


ENABLED           = _b("WATCHDOG_ENABLED", "0")
INTERVAL_SEC      = _f("WATCHDOG_INTERVAL_SEC", 300.0)
TARGET_BASE       = os.getenv("WATCHDOG_TARGET_BASE",
                              "https://kalshiml-production.up.railway.app").rstrip("/")
HTTP_TIMEOUT      = _f("WATCHDOG_HTTP_TIMEOUT_SEC", 20.0)
LOG_TAIL_N        = int(_f("WATCHDOG_LOG_TAIL_N", 2000))

RESEND_API_KEY    = os.getenv("RESEND_API_KEY", "").strip()
ALERT_TO          = [e.strip() for e in os.getenv("ALERT_EMAIL_TO", "").split(",") if e.strip()]
ALERT_FROM        = os.getenv("ALERT_EMAIL_FROM", "").strip()
RESEND_URL        = "https://api.resend.com/emails"
MAX_EMAILS_PER_HR = int(_f("WATCHDOG_MAX_EMAILS_PER_HOUR", 10))

# thresholds
RESTART_WINDOW_SEC   = _f("WATCHDOG_RESTART_WINDOW_SEC", 1800.0)
RESTART_MAX          = int(_f("WATCHDOG_RESTART_MAX", 3))
NO_CANDIDATE_HOURS   = _f("WATCHDOG_NO_CANDIDATE_HOURS", 6.0)
OPENMETEO_CAP_HOUR   = _f("WATCHDOG_OPENMETEO_CAP_HOUR_UTC", 6.0)
IEM_PIN_SEC          = _f("WATCHDOG_IEM_PIN_SEC", 900.0)
TREND_WINDOW_SEC     = _f("WATCHDOG_TREND_WINDOW_SEC", 6 * 3600.0)
TREND_MIN_SAMPLES    = int(_f("WATCHDOG_TREND_MIN_SAMPLES", 12))
TREND_WORSE_FRAC     = _f("WATCHDOG_TREND_WORSE_FRAC", 0.5)

# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #
_RE_CANDIDATES = re.compile(r"\[SCAN\]\s+(\d+)\s+candidates this cycle")
_RE_DRYRUN     = re.compile(r"DRY_RUN=(\w+)\s+PLACE_REAL_ORDERS=(\w+)")
_RE_CASH       = re.compile(r"live_cash=\$([0-9.]+)")
_RE_IEM_BACKOFF= re.compile(r"iem asos: HTTP (\d+) -- backing off ([0-9.]+)s")


def _is_openmeteo_429(line: str) -> bool:
    """Open-Meteo 429s only. The IEM feed also 429s and has its own limit and
    its own check — conflating them would fire the quota alert on an obs
    problem, which is how you learn to ignore an alert."""
    if "429" not in line:
        return False
    low = line.lower()
    if "iem asos" in low:
        return False
    return "open-meteo" in low or "openmeteo" in low or "api.open-meteo.com" in low


def parse_log_lines(lines: list[str]) -> dict[str, Any]:
    """Derive per-cycle facts from a dashboard log tail. Pure — unit-testable
    without network."""
    candidates: list[int] = []
    dry_run: Optional[bool] = None
    real_orders: Optional[bool] = None
    cash: Optional[float] = None
    om_429 = 0
    iem_backoff_max = 0.0

    for ln in lines:
        m = _RE_CANDIDATES.search(ln)
        if m:
            candidates.append(int(m.group(1)))
        m = _RE_DRYRUN.search(ln)
        if m:
            dry_run = m.group(1).strip().lower() in ("true", "1", "yes")
            real_orders = m.group(2).strip().lower() in ("true", "1", "yes")
        m = _RE_CASH.search(ln)
        if m:
            try:
                cash = float(m.group(1))
            except ValueError:
                pass
        if _is_openmeteo_429(ln):
            om_429 += 1
        m = _RE_IEM_BACKOFF.search(ln)
        if m:
            try:
                iem_backoff_max = max(iem_backoff_max, float(m.group(2)))
            except ValueError:
                pass

    return {
        "scan_cycles": len(candidates),
        "candidates_total": sum(candidates),
        "candidates_last": (candidates[-1] if candidates else None),
        "dry_run": dry_run,
        "place_real_orders": real_orders,
        "live_cash": cash,
        "openmeteo_429": om_429,
        "iem_backoff_max_sec": iem_backoff_max,
    }


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
_STATE: dict[str, Any] = {
    "started_at": time.time(),
    "last_cycle_at": None,
    "last_cycle_ok": None,
    "last_error": None,
    "cycles": 0,
    "checks": {},            # name -> {"bad": bool, "since": epoch, "msg": str}
    "last_sample": {},
    "emails_sent": 0,
    "emails_suppressed": 0,
    "last_email_at": None,
}
_SAMPLES: deque = deque(maxlen=1200)     # rolling trend window
_PROC_STARTS: deque = deque(maxlen=50)   # (epoch_seen, proc_started_string)
_EMAIL_TIMES: deque = deque(maxlen=200)
_FIRST_OM429_BY_DAY: dict[str, float] = {}   # 'YYYY-MM-DD' -> UTC hour first seen


def _utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _utc_hour(ts: float) -> float:
    d = datetime.fromtimestamp(ts, timezone.utc)
    return d.hour + d.minute / 60.0


# --------------------------------------------------------------------------- #
# Checks — each returns (bad: bool, message: str)
# --------------------------------------------------------------------------- #
def _chk_unreachable(s: dict, now: float) -> tuple[bool, str]:
    if s.get("reachable"):
        return False, "dashboard reachable"
    return True, f"dashboard unreachable: {s.get('fetch_error') or 'unknown error'}"


def _chk_restart_loop(s: dict, now: float) -> tuple[bool, str]:
    recent = [p for (t, p) in _PROC_STARTS if now - t <= RESTART_WINDOW_SEC]
    distinct = len(set(recent))
    if distinct >= RESTART_MAX:
        return True, (f"{distinct} distinct proc_started values in the last "
                      f"{RESTART_WINDOW_SEC/60:.0f}m — restart loop")
    return False, f"{distinct} restart(s) in window"


def _chk_safety_flags(s: dict, now: float) -> tuple[bool, str]:
    """Canary. Should never fire. If it does, real money is in play and that is
    worth an email even at 3am."""
    dr = s.get("dry_run")
    ro = s.get("place_real_orders")
    if dr is None and ro is None:
        return False, "flags not seen in window"
    if dr is False or ro is True:
        return True, f"SAFETY: DRY_RUN={dr} PLACE_REAL_ORDERS={ro} — live orders possible"
    return False, f"DRY_RUN={dr} PLACE_REAL_ORDERS={ro}"


def _chk_no_candidates(s: dict, now: float) -> tuple[bool, str]:
    """Zero candidates is normal for a cycle and pathological for a day. Only
    fires once it has been continuously zero across the configured span, so a
    quiet afternoon does not wake anyone."""
    window = [x for x in _SAMPLES if now - x["t"] <= NO_CANDIDATE_HOURS * 3600]
    scanned = [x for x in window if x.get("scan_cycles")]
    if len(scanned) < TREND_MIN_SAMPLES:
        return False, "not enough scan history yet"
    span_h = (now - scanned[0]["t"]) / 3600.0
    if span_h < NO_CANDIDATE_HOURS * 0.8:
        return False, f"only {span_h:.1f}h of history"
    if all((x.get("candidates_total") or 0) == 0 for x in scanned):
        return True, (f"0 candidates across every scan for {span_h:.1f}h — "
                      f"the system is running but learning nothing")
    return False, "candidates present"


def _chk_openmeteo_cap(s: dict, now: float) -> tuple[bool, str]:
    day = _utc_day(now)
    first = _FIRST_OM429_BY_DAY.get(day)
    if first is None:
        return False, "no Open-Meteo 429 yet today"
    if first < OPENMETEO_CAP_HOUR:
        return True, (f"Open-Meteo daily cap hit at {first:.1f}h UTC "
                      f"(bar is {OPENMETEO_CAP_HOUR:.1f}h) — forecast recorder "
                      f"starves for the rest of the day")
    return False, f"first 429 at {first:.1f}h UTC — acceptable"


def _chk_iem_pinned(s: dict, now: float) -> tuple[bool, str]:
    v = s.get("iem_backoff_max_sec") or 0.0
    if v >= IEM_PIN_SEC:
        return True, (f"IEM breaker pinned at {v:.0f}s cap — sweep cadence, not "
                      f"retry policy, is the binding constraint")
    return False, f"IEM backoff max {v:.0f}s"


def _chk_degradation(s: dict, now: float) -> tuple[bool, str]:
    """Slow-trend check: compare the recent window against the one before it.
    In-memory only, so it stays quiet for a while after each redeploy."""
    w = TREND_WINDOW_SEC
    recent = [x for x in _SAMPLES if now - x["t"] <= w]
    prior  = [x for x in _SAMPLES if w < now - x["t"] <= 2 * w]
    if len(recent) < TREND_MIN_SAMPLES or len(prior) < TREND_MIN_SAMPLES:
        return False, "not enough history for a trend"

    def rate(rows, key):
        return sum((r.get(key) or 0) for r in rows) / max(1, len(rows))

    r_cand, p_cand = rate(recent, "candidates_total"), rate(prior, "candidates_total")
    r_429,  p_429  = rate(recent, "openmeteo_429"),   rate(prior, "openmeteo_429")

    problems = []
    if p_cand > 0 and r_cand < p_cand * (1 - TREND_WORSE_FRAC):
        problems.append(f"candidate rate {p_cand:.1f} -> {r_cand:.1f} per cycle")
    if r_429 > p_429 + 1 and r_429 > p_429 * (1 + TREND_WORSE_FRAC):
        problems.append(f"Open-Meteo 429 rate {p_429:.1f} -> {r_429:.1f} per cycle")
    if problems:
        return True, ("degrading over the last "
                      f"{w/3600:.0f}h vs the previous {w/3600:.0f}h: "
                      + "; ".join(problems))
    return False, "trends stable"


CHECKS = {
    "dashboard_unreachable": _chk_unreachable,
    "restart_loop":          _chk_restart_loop,
    "safety_flags":          _chk_safety_flags,
    "no_candidates":         _chk_no_candidates,
    "openmeteo_cap_early":   _chk_openmeteo_cap,
    "iem_backoff_pinned":    _chk_iem_pinned,
    "degradation":           _chk_degradation,
}

# Checks that only make sense when we actually reached the dashboard.
_NEEDS_REACHABLE = set(CHECKS) - {"dashboard_unreachable"}


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
async def sample_once(client: httpx.AsyncClient) -> dict[str, Any]:
    now = time.time()
    out: dict[str, Any] = {"t": now, "reachable": False, "fetch_error": None}
    try:
        vr = await client.get(f"{TARGET_BASE}/api/version", timeout=HTTP_TIMEOUT)
        vr.raise_for_status()
        v = vr.json()
        out["commit"] = v.get("commit")
        out["proc_started"] = v.get("proc_started")
        out["reachable"] = True
    except Exception as e:
        out["fetch_error"] = f"{type(e).__name__}: {e}"
        return out

    try:
        lr = await client.get(f"{TARGET_BASE}/api/logs", params={"n": LOG_TAIL_N},
                              timeout=HTTP_TIMEOUT)
        lr.raise_for_status()
        payload = lr.json()
        raw = payload.get("lines") or []
        lines = [(x.get("line") if isinstance(x, dict) else str(x)) for x in raw]
        out.update(parse_log_lines([l for l in lines if l]))
    except Exception as e:
        # Version worked, logs did not: still "reachable", just thinner data.
        out["log_error"] = f"{type(e).__name__}: {e}"
    return out


def record_sample(s: dict[str, Any]) -> None:
    now = s["t"]
    _SAMPLES.append(s)
    if s.get("proc_started"):
        if not _PROC_STARTS or _PROC_STARTS[-1][1] != s["proc_started"]:
            _PROC_STARTS.append((now, s["proc_started"]))
    if (s.get("openmeteo_429") or 0) > 0:
        day = _utc_day(now)
        _FIRST_OM429_BY_DAY.setdefault(day, _utc_hour(now))
        for k in list(_FIRST_OM429_BY_DAY):
            if k < _utc_day(now - 7 * 86400):
                _FIRST_OM429_BY_DAY.pop(k, None)


def evaluate(s: dict[str, Any], now: Optional[float] = None) -> list[dict[str, Any]]:
    """Run checks, update state, return the list of TRANSITIONS to report."""
    now = now if now is not None else time.time()
    transitions = []
    for name, fn in CHECKS.items():
        if not s.get("reachable") and name in _NEEDS_REACHABLE:
            continue   # don't cascade: one outage should not fire six alerts
        try:
            bad, msg = fn(s, now)
        except Exception as e:
            bad, msg = False, f"check error: {type(e).__name__}: {e}"
        prev = _STATE["checks"].get(name) or {"bad": False, "since": now, "msg": ""}
        if bad != prev["bad"]:
            _STATE["checks"][name] = {"bad": bad, "since": now, "msg": msg}
            transitions.append({"check": name, "bad": bad, "msg": msg,
                                "was_for_sec": round(now - prev["since"], 1)})
        else:
            prev["msg"] = msg
            _STATE["checks"][name] = prev
    return transitions


# --------------------------------------------------------------------------- #
# Delivery (Resend)
# --------------------------------------------------------------------------- #
def email_configured() -> bool:
    return bool(RESEND_API_KEY and ALERT_TO and ALERT_FROM)


def _rate_limited(now: float) -> bool:
    while _EMAIL_TIMES and now - _EMAIL_TIMES[0] > 3600:
        _EMAIL_TIMES.popleft()
    return len(_EMAIL_TIMES) >= MAX_EMAILS_PER_HR


async def send_email(client: httpx.AsyncClient, subject: str, body: str) -> dict[str, Any]:
    now = time.time()
    if not email_configured():
        return {"sent": False, "reason": "email not configured"}
    if _rate_limited(now):
        _STATE["emails_suppressed"] += 1
        return {"sent": False, "reason": "rate limited"}
    try:
        r = await client.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {RESEND_API_KEY}",
                     "Content-Type": "application/json"},
            json={"from": ALERT_FROM, "to": ALERT_TO,
                  "subject": subject, "text": body},
            timeout=HTTP_TIMEOUT)
        ok = r.status_code < 300
        if ok:
            _EMAIL_TIMES.append(now)
            _STATE["emails_sent"] += 1
            _STATE["last_email_at"] = now
        return {"sent": ok, "status": r.status_code,
                "detail": (None if ok else r.text[:300])}
    except Exception as e:
        return {"sent": False, "reason": f"{type(e).__name__}: {e}"}


def format_alert(transitions: list[dict], s: dict) -> tuple[str, str]:
    bad = [t for t in transitions if t["bad"]]
    good = [t for t in transitions if not t["bad"]]
    if bad:
        head = bad[0]["check"] if len(bad) == 1 else f"{len(bad)} checks"
        subject = f"[KML] {head} — attention"
    else:
        head = good[0]["check"] if len(good) == 1 else f"{len(good)} checks"
        subject = f"[KML] recovered: {head}"

    lines = []
    for t in bad:
        lines.append(f"PROBLEM  {t['check']}\n  {t['msg']}")
    for t in good:
        lines.append(f"RECOVERED  {t['check']} (was bad for "
                     f"{t['was_for_sec']/60:.0f}m)\n  {t['msg']}")
    lines.append("")
    lines.append("Context at detection:")
    for k in ("commit", "proc_started", "scan_cycles", "candidates_total",
              "openmeteo_429", "iem_backoff_max_sec", "dry_run",
              "place_real_orders", "live_cash"):
        if k in s:
            lines.append(f"  {k}: {s[k]}")
    lines.append("")
    lines.append(f"target: {TARGET_BASE}")
    lines.append(f"at: {datetime.now(timezone.utc).isoformat()}")
    return subject, "\n".join(lines)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
async def run_cycle(client: httpx.AsyncClient) -> dict[str, Any]:
    s = await sample_once(client)
    record_sample(s)
    transitions = evaluate(s)
    _STATE["last_sample"] = s
    _STATE["last_cycle_at"] = s["t"]
    _STATE["cycles"] += 1
    result = {"sample": s, "transitions": transitions, "email": None}
    if transitions:
        subject, body = format_alert(transitions, s)
        result["email"] = await send_email(client, subject, body)
    return result


async def watchdog_worker() -> None:
    """Background task. Never raises out; a bad cycle is recorded and retried."""
    if not ENABLED:
        _STATE["last_error"] = "disabled (WATCHDOG_ENABLED unset)"
        return
    import asyncio
    async with httpx.AsyncClient(follow_redirects=False) as client:
        while True:
            try:
                await run_cycle(client)
                _STATE["last_cycle_ok"] = True
                _STATE["last_error"] = None
            except Exception as e:
                _STATE["last_cycle_ok"] = False
                _STATE["last_error"] = f"{type(e).__name__}: {e}"
            await asyncio.sleep(INTERVAL_SEC)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
watchdog_router = APIRouter(prefix="/watchdog", tags=["watchdog"])


@watchdog_router.get("/status", summary="Watchdog state and current check results")
async def watchdog_status() -> dict[str, Any]:
    now = time.time()
    return {
        "enabled": ENABLED,
        "target": TARGET_BASE,
        "interval_sec": INTERVAL_SEC,
        "email_configured": email_configured(),          # never the key itself
        "email_to_count": len(ALERT_TO),
        "cycles": _STATE["cycles"],
        "last_cycle_at": _STATE["last_cycle_at"],
        "last_cycle_age_sec": (None if not _STATE["last_cycle_at"]
                               else round(now - _STATE["last_cycle_at"], 1)),
        "last_cycle_ok": _STATE["last_cycle_ok"],
        "last_error": _STATE["last_error"],
        "emails_sent": _STATE["emails_sent"],
        "emails_suppressed": _STATE["emails_suppressed"],
        "samples_held": len(_SAMPLES),
        "first_openmeteo_429_utc_hour_by_day": dict(_FIRST_OM429_BY_DAY),
        "checks": {k: {"bad": v["bad"], "msg": v["msg"],
                       "since_sec": round(now - v["since"], 1)}
                   for k, v in _STATE["checks"].items()},
        "last_sample": _STATE["last_sample"],
    }


@watchdog_router.post("/check", summary="Run one watchdog cycle now")
async def watchdog_check() -> dict[str, Any]:
    async with httpx.AsyncClient(follow_redirects=False) as client:
        return await run_cycle(client)


@watchdog_router.post("/test_email", summary="Send a test alert email via Resend")
async def watchdog_test_email() -> dict[str, Any]:
    if not email_configured():
        return {"sent": False,
                "reason": "set RESEND_API_KEY, ALERT_EMAIL_FROM, ALERT_EMAIL_TO"}
    async with httpx.AsyncClient(follow_redirects=False) as client:
        return await send_email(
            client,
            "[KML] watchdog test",
            "Test alert from the KalshiML watchdog.\n\n"
            f"target: {TARGET_BASE}\n"
            f"interval: {INTERVAL_SEC:.0f}s\n"
            f"at: {datetime.now(timezone.utc).isoformat()}\n\n"
            "If you got this, delivery works. Real alerts fire only on state "
            "changes, so silence from here on is the good outcome.")
