"""Success judge for the booking-alert task.

Primary path uses an OpenAI model to fuzzily verify that a Slack alert is a
well-formed "New Booking Alert" for a specific booking row. If the API key or
network is unavailable (or any call fails), it degrades gracefully to a
deterministic substring/heuristic check so the environment never hard-depends
on network access.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

_DOTENV_LOADED = False


def _load_env() -> None:
    """Load OPENAI_API_KEY from the workspace-root .env using an absolute path."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    if os.environ.get("OPENAI_API_KEY"):
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    here = Path(__file__).resolve()
    candidates = [p / ".env" for p in here.parents]
    for env_path in candidates:
        if env_path.is_file():
            try:
                load_dotenv(str(env_path))
            except Exception:
                pass
            if os.environ.get("OPENAI_API_KEY"):
                break


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _humanized_time_parts(value: str) -> list[str]:
    """Return substrings that a faithful alert might contain for a timestamp."""
    parts: list[str] = [value]
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})", value or "")
    if m:
        y, mo, d, hh, mm = m.groups()
        parts.append(f"{y}-{mo}-{d}")
        parts.append(f"{hh}:{mm}")
        h = int(hh)
        ampm = "am" if h < 12 else "pm"
        h12 = h % 12 or 12
        parts.append(f"{h12}:{mm}")
        parts.append(f"{h12}:{mm} {ampm}")
    return [p for p in parts if p]


def deterministic_judge(
    alert_text: str,
    row: dict[str, str],
    columns: list[str],
    column_roles: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Heuristic check that the alert contains all required booking fields."""
    roles = column_roles or {}
    text_n = _norm(alert_text)

    def col_for(role: str, default_idx: int) -> Optional[str]:
        if role in roles:
            return roles[role]
        return columns[default_idx] if default_idx < len(columns) else None

    name_c = col_for("guest_name", 1)
    exp_c = col_for("experience", 2)
    boat_c = col_for("boat", 3)
    count_c = col_for("guest_count", 4)
    time_c = col_for("start_time", 5)

    checks: dict[str, bool] = {}
    checks["title"] = "new booking alert" in text_n
    if name_c:
        checks["guest_name"] = _norm(row.get(name_c, "")) in text_n and bool(row.get(name_c))
    if count_c:
        cv = _norm(row.get(count_c, ""))
        checks["guest_count"] = bool(cv) and re.search(rf"\b{re.escape(cv)}\b", text_n) is not None
    if exp_c:
        checks["experience"] = _norm(row.get(exp_c, "")) in text_n and bool(row.get(exp_c))
    if boat_c:
        checks["boat"] = _norm(row.get(boat_c, "")) in text_n and bool(row.get(boat_c))
    if time_c:
        opts = [_norm(p) for p in _humanized_time_parts(row.get(time_c, ""))]
        checks["start_time"] = any(o and o in text_n for o in opts)
    checks["prepare_instruction"] = ("prepare" in text_n) and (
        "crew" in text_n or "boat" in text_n
    )

    required = [k for k in checks if k != "title"]
    passed = sum(1 for k in required if checks[k])
    score = passed / max(1, len(required))
    correct = all(checks[k] for k in required) and checks.get("title", False)
    missing = [k for k, v in checks.items() if not v]
    return {
        "correct": correct,
        "score": round(score, 3),
        "checks": checks,
        "missing": missing,
        "method": "deterministic",
    }


def llm_judge(
    alert_text: str,
    row: dict[str, str],
    columns: list[str],
    model: str = "gpt-4o-mini",
) -> Optional[dict[str, Any]]:
    """Ask an OpenAI model to verify the alert. Returns None on any failure."""
    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        booking = {c: row.get(c, "") for c in columns}
        sys_prompt = (
            "You validate Slack alerts for a boat-tour booking automation. "
            "Given the booking record and the posted Slack message, decide if the "
            "message is a correctly formatted 'New Booking Alert' that conveys this "
            "specific booking. It must clearly include: the guest name, the group "
            "size / guest count, the experience type, the boat/vessel assignment, "
            "and the scheduled start time, plus an instruction to prepare the boat "
            "and crew. Minor wording/formatting differences are fine; the data must "
            "match the booking. Respond ONLY with JSON: "
            '{"correct": bool, "score": number 0..1, "missing": [string], "reason": string}.'
        )
        user_prompt = (
            f"BOOKING RECORD (JSON):\n{json.dumps(booking, indent=2)}\n\n"
            f"SLACK MESSAGE:\n\"\"\"\n{alert_text}\n\"\"\""
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "correct": bool(data.get("correct")),
            "score": float(data.get("score", 1.0 if data.get("correct") else 0.0)),
            "missing": data.get("missing", []),
            "reason": data.get("reason", ""),
            "method": f"openai:{model}",
        }
    except Exception as exc:  # network, auth, quota, parsing - all fall back
        return {"error": str(exc), "method": "openai-failed"}


def judge_alert(
    alert_text: str,
    row: dict[str, str],
    columns: list[str],
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
    column_roles: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Judge an alert, preferring the LLM and falling back deterministically.

    The deterministic result is always computed (cheap, offline) and returned as
    a cross-check alongside the chosen verdict.
    """
    det = deterministic_judge(alert_text, row, columns, column_roles)
    result: dict[str, Any] = {"deterministic": det}

    llm: Optional[dict[str, Any]] = None
    if use_llm:
        llm = llm_judge(alert_text, row, columns, model=model)

    if llm and "error" not in llm:
        result.update(
            {
                "correct": bool(llm["correct"]),
                "score": float(llm.get("score", 0.0)),
                "method": llm["method"],
                "reason": llm.get("reason", ""),
                "missing": llm.get("missing", []),
            }
        )
    else:
        result.update(
            {
                "correct": det["correct"],
                "score": det["score"],
                "method": det["method"],
                "reason": "deterministic fallback" + (
                    f" (llm error: {llm['error']})" if llm and "error" in llm else ""
                ),
                "missing": det["missing"],
            }
        )
    return result
