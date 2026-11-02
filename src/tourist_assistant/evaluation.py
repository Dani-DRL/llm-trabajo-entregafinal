"""Evaluación: recuperación (hit@k, MRR) y conversaciones completas con comprobaciones deterministas"""

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

from tourist_assistant.config import get_settings


def evaluate_retrieval(store, questions_path: Path, k: int = 4) -> dict:
    """Devuelve hit@1, hit@3, MRR, latencia media (ms) y las preguntas falladas"""
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    ranks, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        hits = store.similarity_search(q["question"], k=k)
        latencies.append(time.perf_counter() - start)
        rank = next(
            (
                i
                for i, doc in enumerate(hits, start=1)
                if any(e.lower() in doc.page_content.lower() for e in q["expected"])
            ),
            None,
        )
        ranks.append(rank)
    n = len(questions)
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "latency_ms": 1000 * sum(latencies) / n,
        "missed": [q["id"] for q, r in zip(questions, ranks) if r is None],
    }


REFUSAL_RE = re.compile(
    r"no tengo|no dispongo|no encuentro|no puedo|solo puedo|lo siento|no figura|no consta"
    r"|no (?:se )?(?:aparece|menciona|incluye|indica|especifica|proporciona|ofrece|detalla|dispone)"
    r"|no hay (?:información|datos)",
    re.IGNORECASE,
)


INVALID_DATE_RE = re.compile(
    r"no existe|no es (?:una )?fecha|fecha (?:no )?v[aá]lida|inv[aá]lid", re.IGNORECASE
)


def _log_lines(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def run_case(ask_fn, case: dict, run_id: str) -> dict:
    """Ejecuta los turnos de un caso; ask_fn(mensaje, thread_id) devuelve (texto, fuentes, ruta)"""
    log_path = get_settings().log_dir / "weather_calls.jsonl"
    result = {"text": "", "sources": [], "route": None, "tool_calls": [], "error": None}
    start = time.perf_counter()
    try:
        for message in case["turns"]:
            before = len(_log_lines(log_path))
            start = time.perf_counter()
            text, sources, route = ask_fn(message, f"{run_id}-{case['id']}")
            result.update(text=text, sources=sources, route=route)
            result["tool_calls"] = _log_lines(log_path)[before:]
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"[:200]
    result["latency_s"] = time.perf_counter() - start
    return result


def check_result(checks: dict, result: dict, today: date | None = None) -> dict[str, bool]:
    """Aplica las comprobaciones de un caso y devuelve {nombre: cumplida}"""
    today = today or date.today()
    text = result["text"]
    lowered = text.lower()
    outcome = {"no_crash": result["error"] is None and bool(text.strip())}
    if "contains" in checks:
        outcome["contains"] = all(
            any(option.lower() in lowered for option in group) for group in checks["contains"]
        )
    if "sources_include" in checks:
        joined = " ".join(result["sources"]).lower()
        outcome["sources_include"] = all(
            item.lower() in joined for item in checks["sources_include"]
        )
    if checks.get("refusal"):
        outcome["refusal"] = bool(REFUSAL_RE.search(text))
    if "forbidden" in checks:
        outcome["forbidden"] = not any(item.lower() in lowered for item in checks["forbidden"])
    if "route" in checks:
        outcome["route"] = result["route"] in checks["route"]
    if "weather_offset" in checks:
        expected = (today + timedelta(days=checks["weather_offset"])).isoformat()
        outcome["weather_offset"] = any(
            call["fecha_solicitada"] == expected and call["resultado"] == "ok"
            for call in result["tool_calls"]
        )
    if checks.get("weather_error"):
        tool_error = any(call["resultado"] != "ok" for call in result["tool_calls"])
        outcome["weather_error"] = tool_error or bool(
            REFUSAL_RE.search(text) or INVALID_DATE_RE.search(text)
        )
    return outcome


def evaluate_e2e(ask_fn, cases_path: Path, run_id: str = "e2e"):
    """Ejecuta los casos y devuelve un DataFrame con un caso por fila"""
    import pandas as pd

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        result = run_case(ask_fn, case, run_id)
        checks = check_result(case["checks"], result)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": all(checks.values()),
                "checks": checks,
                "route": result["route"],
                "latency_s": round(result["latency_s"], 1),
                "text": result["text"],
                "sources": result["sources"],
                "tool_called": bool(result["tool_calls"]),
                "error": result["error"],
            }
        )
        mark = "ok" if rows[-1]["passed"] else "FALLA"
        print(f"{case['id']} {mark} [{result['route']}] {result['latency_s']:.0f}s", flush=True)
    return pd.DataFrame(rows)
