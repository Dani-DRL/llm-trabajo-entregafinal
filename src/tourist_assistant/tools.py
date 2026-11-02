"""Herramienta get_weather: predicción del tiempo en Tenerife (Open-Meteo, con fallback simulado)"""

import hashlib
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tourist_assistant.config import get_settings

API_URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE, LONGITUDE = 28.4636, -16.2518  # Santa Cruz de Tenerife
MAX_DAYS_AHEAD = 15  # límite de la API de predicción de Open-Meteo
WMO_DESCRIPTIONS = {
    0: "despejado", 1: "mayormente despejado", 2: "parcialmente nublado", 3: "nublado",
    45: "niebla", 48: "niebla con escarcha", 51: "llovizna ligera", 53: "llovizna",
    55: "llovizna intensa", 61: "lluvia ligera", 63: "lluvia", 65: "lluvia intensa",
    80: "chubascos ligeros", 81: "chubascos", 82: "chubascos violentos", 95: "tormenta",
}  # fmt: skip


WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def calendar_context(today: date | None = None) -> str:
    """Fecha de hoy y calendario de los próximos días, para que el modelo no tenga que contar"""
    today = today or date.today()
    days = [today + timedelta(days=n) for n in range(MAX_DAYS_AHEAD + 1)]
    labels = {0: "hoy", 1: "mañana", 2: "pasado mañana"}
    listing = ", ".join(
        f"{WEEKDAYS[d.weekday()]} {d.isoformat()} ({labels.get(n, f'en {n} días')})"
        for n, d in enumerate(days)
    )
    return f"Hoy es {WEEKDAYS[today.weekday()]} {today.isoformat()}. Calendario: {listing}."


def weather_system_prompt(today: date | None = None) -> str:
    return (
        "Eres un asistente turístico de Tenerife. "
        f"{calendar_context(today)} "
        "Ante CUALQUIER pregunta sobre el tiempo llama siempre a la herramienta get_weather "
        "con la fecha en formato AAAA-MM-DD, aunque la fecha te parezca pasada o lejana: la "
        "herramienta valida las fechas. Nunca respondas sobre el tiempo sin llamarla. Si "
        "devuelve un error, explícaselo al usuario con claridad."
    )


class WeatherInput(BaseModel):
    """Argumentos de get_weather"""

    fecha: str = Field(
        description="Fecha de la predicción en formato AAAA-MM-DD, p. ej. 2026-10-25"
    )


def _error(code: str, message: str) -> dict:
    return {"error": code, "mensaje": message}


def _validate_date(fecha: str, today: date) -> tuple[date | None, dict | None]:
    """Convierte la fecha o devuelve un error (formato, pasada o fuera de rango)"""
    try:
        day = datetime.strptime(fecha.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None, _error("fecha_invalida", f"'{fecha}' no tiene formato AAAA-MM-DD.")
    if day < today:
        return None, _error(
            "fecha_pasada", f"{day} ya ha pasado; solo hay predicción, no histórico."
        )
    last = today + timedelta(days=MAX_DAYS_AHEAD)
    if day > last:
        return None, _error(
            "fecha_fuera_de_rango",
            f"Solo hay predicción hasta {last} ({MAX_DAYS_AHEAD} días desde hoy).",
        )
    return day, None


def _fetch_open_meteo(day: date, client: httpx.Client) -> dict:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Atlantic/Canary",
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
    }
    response = client.get(API_URL, params=params, timeout=5.0)
    response.raise_for_status()
    daily = response.json()["daily"]
    return {
        "fecha": day.isoformat(),
        "temp_max_c": daily["temperature_2m_max"][0],
        "temp_min_c": daily["temperature_2m_min"][0],
        "precipitacion_mm": daily["precipitation_sum"][0],
        "descripcion": WMO_DESCRIPTIONS.get(daily["weather_code"][0], "sin descripción"),
        "fuente": "open-meteo",
    }


def _simulate(day: date) -> dict:
    """Predicción simulada (misma fecha, mismo resultado) para el fallback"""
    seed = int(hashlib.md5(day.isoformat().encode()).hexdigest(), 16)
    t_max = 21 + seed % 7
    return {
        "fecha": day.isoformat(),
        "temp_max_c": float(t_max),
        "temp_min_c": float(t_max - 4),
        "precipitacion_mm": float(seed // 7 % 3),
        "descripcion": ["despejado", "parcialmente nublado", "nublado"][seed // 3 % 3],
        "fuente": "simulado",
    }


def _log_call(log_dir: Path, record: dict) -> None:
    """Añade una línea al registro de llamadas (una por intento)"""
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
    with (log_dir / "weather_calls.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_weather(
    fecha: str,
    *,
    today: date | None = None,
    client: httpx.Client | None = None,
    log_dir: Path | None = None,
    allow_fallback: bool = True,
) -> dict:
    """Predicción del tiempo en Tenerife para una fecha (AAAA-MM-DD)

    No lanza excepciones: devuelve el tiempo o un dict {"error", "mensaje"}. Si Open-Meteo
    falla, usa datos simulados. Cada intento se registra en el log.
    """
    today = today or date.today()
    log_dir = log_dir or get_settings().log_dir
    start = time.perf_counter()
    detail = ""

    day, result = _validate_date(fecha, today)
    if day is not None:
        try:
            http = client or httpx.Client()
            try:
                result = _fetch_open_meteo(day, http)
            finally:
                if client is None:
                    http.close()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            detail = f"{type(error).__name__}: {error}"
            result = (
                _simulate(day)
                if allow_fallback
                else _error(
                    "servicio_no_disponible", "No se pudo consultar el servicio del tiempo."
                )
            )

    _log_call(
        log_dir,
        {
            "fecha_solicitada": fecha,
            "resultado": result.get("error", "ok"),
            "fuente": result.get("fuente"),
            "detalle": detail,
            "latencia_ms": round(1000 * (time.perf_counter() - start)),
        },
    )
    return result


weather_tool = StructuredTool.from_function(
    func=lambda fecha: get_weather(fecha),
    name="get_weather",
    description=(
        "Devuelve la predicción del tiempo en Tenerife para una fecha concreta "
        "(máx/mín en °C, lluvia y descripción). Solo fechas desde hoy hasta 15 días."
    ),
    args_schema=WeatherInput,
)
