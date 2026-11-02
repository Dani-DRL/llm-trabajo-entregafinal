"""Genera docs/INFORME.pdf a partir de docs/INFORME.md (Markdown → HTML → PDF con Chrome sin ventana)

Uso, desde code/:   .venv/bin/python docs/build_pdf.py
Requiere Google Chrome (ruta por defecto de macOS; otra con la variable CHROME_PATH) y markdown-it-py.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

DOCS = Path(__file__).resolve().parent
CHROME = os.environ.get(
    "CHROME_PATH", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; } tr { break-inside: avoid; }
th, td { border: 1px solid #c9c9c9; padding: 3pt 5pt; vertical-align: top; text-align: left; } th { background: #f0f0f0; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6pt; font-size: 8.4pt; line-height: 1.3; white-space: pre-wrap; break-inside: avoid; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 2pt; } pre code { background: none; padding: 0; }
img { max-width: 82%; display: block; margin: 8pt auto; break-inside: avoid; } hr { border: 0; border-top: 1px solid #ccc; margin: 10pt 0; }
li { margin: 1.5pt 0; } p { margin: 5pt 0; }
"""


def main() -> None:
    source, output = DOCS / "INFORME.md", DOCS / "INFORME.pdf"
    body = MarkdownIt("commonmark").enable("table").render(source.read_text(encoding="utf-8"))
    html = (
        f'<!doctype html><html lang="es"><head><meta charset="utf-8"><base href="file://{DOCS}/">'
        f"<title>Informe final</title><style>{CSS}</style></head><body>{body}</body></html>"
    )
    with tempfile.TemporaryDirectory() as folder:
        page = Path(folder) / "informe.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={output}", f"file://{page}"],
            check=True, capture_output=True, timeout=120,
        )  # fmt: skip
    print(f"{output} ({output.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
