"""Ejecuta la evaluación y guarda los resultados en eval/results/

Uso, desde code/ (con Ollama en marcha para los embeddings):
    .venv/bin/python eval/run_e2e.py graph [N]    # N ejecuciones (3 por defecto)
"""

import json
import sys
import warnings
from datetime import date

import pandas as pd

from tourist_assistant.config import PROJECT_ROOT, get_settings
from tourist_assistant.evaluation import evaluate_e2e
from tourist_assistant.graph import ask, build_graph
from tourist_assistant.llm import get_llm
from tourist_assistant.retrieval import get_index

warnings.filterwarnings("ignore")
EVAL_DIR = PROJECT_ROOT / "eval"


def main(system: str, runs: int = 3) -> None:
    settings = get_settings()
    llm = get_llm(settings)
    if system != "graph":
        raise SystemExit(f"Sistema desconocido: {system}")
    graph = build_graph(get_index(settings), llm, settings.max_history_tokens)

    def ask_fn(message, thread_id):
        return ask(graph, message, thread_id)

    frames = []
    for run in range(1, runs + 1):
        print(f"--- ejecución {run}/{runs} ---", flush=True)
        frame = evaluate_e2e(ask_fn, EVAL_DIR / "e2e_cases.json", run_id=f"{system}-r{run}")
        frames.append(frame.assign(run=run))
        print(f"acierto de la ejecución {run}: {frame.passed.mean():.2f}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    out = {
        "system": system,
        "date": date.today().isoformat(),
        "model": settings.llm_model,
        "params": {"temperature": settings.temperature, "top_p": settings.top_p},
        "rows": df.to_dict(orient="records"),
    }
    model_slug = settings.llm_model.replace(":", "-").replace("/", "-")
    path = EVAL_DIR / "results" / f"e2e_{system}_{model_slug}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    per_run = df.groupby("run").passed.mean()
    print(f"\nAcierto por ejecución: {per_run.round(2).tolist()} → {path}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "graph", int(sys.argv[2]) if len(sys.argv) > 2 else 3
    )
