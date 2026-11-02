"""Ingesta: PDF → chunks con zona, lugar y páginas como metadatos"""

import re
from pathlib import Path
from typing import NamedTuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

_MARKERS = {"-", "o", "•", "▪"}
_ZONE_RE = re.compile(r"^ZONA\s+(\w+)$")
_PLACE_RE = re.compile(r"^•\s*(.+)$")
_SUBITEM_RE = re.compile(r"^▪\s*(.+)$")
_BULLET_RE = re.compile(r"^[-o•▪]\s")
_LINK_RE = re.compile(
    r"\[(?:vídeo|ubicación|web|Tripadvisor)(?:\s*-\s*ubicación)?\]"
    r"|\(ubicación\)|[-–]\s*ubicación\s*[-–]?"
)


_SPACES_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" ([.,])")


class Line(NamedTuple):
    """Una línea del PDF y su página"""

    text: str
    page: int


def load_pdf_lines(path: Path) -> list[Line]:
    """Extrae las líneas del PDF; una viñeta suelta se une a la línea siguiente"""
    lines: list[Line] = []
    marker = ""
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        for raw in (page.extract_text() or "").splitlines():
            text = raw.strip()
            if not text:
                continue
            if text in _MARKERS:
                marker = text
                continue
            lines.append(Line(f"{marker} {text}" if marker else text, page_number))
            marker = ""
    return lines


def _sub(pattern: re.Pattern, repl: str, text: str, pages: list[int]) -> tuple[str, list[int]]:
    """re.sub que mantiene alineada la lista de páginas"""
    out, out_pages, last = "", [], 0
    for match in pattern.finditer(text):
        out += text[last : match.start()]
        out_pages += pages[last : match.start()]
        replacement = match.expand(repl)
        out += replacement
        out_pages += [pages[match.start()]] * len(replacement)
        last = match.end()
    return out + text[last:], out_pages + pages[last:]


def _clean_pages(text: str, pages: list[int]) -> tuple[str, list[int]]:
    """Quita los marcadores de enlace ([vídeo], (ubicación)…) y los espacios sobrantes"""
    for pattern, repl in ((_LINK_RE, ""), (_SPACES_RE, " "), (_SPACE_BEFORE_PUNCT_RE, r"\1")):
        text, pages = _sub(pattern, repl, text, pages)
    return text, pages


def _clean(text: str) -> str:
    return _clean_pages(text, [0] * len(text))[0].strip()


def _documents(header: str, lines: list[Line], zone: str, place: str, piece_size: int | None):
    """Crea los Document de un lugar (en trozos si piece_size está definido)

    Se guarda la página de cada carácter para citar la página donde está el texto de cada trozo.
    """
    text, page_at = "", []
    for line in lines:
        separator = "\n" if not text or _BULLET_RE.match(line.text) else " "
        text += separator + line.text
        page_at.extend([line.page] * (len(separator) + len(line.text)))
    text, page_at = _clean_pages(text, page_at)
    stripped = text.strip().removeprefix("• ")
    page_at = page_at[len(text) - len(text.lstrip()) + (len(text.strip()) - len(stripped)) :]
    text = stripped

    if piece_size is None:
        pieces = [(text, 0)]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=piece_size, chunk_overlap=piece_size // 8, add_start_index=True
        )
        pieces = [
            (d.page_content, d.metadata["start_index"]) for d in splitter.create_documents([text])
        ]
    return [
        Document(
            page_content=f"[{header}]\n{content}",
            metadata={
                "zone": zone,
                "place": place,
                "page_start": page_at[start],
                "page_end": page_at[min(start + len(content), len(page_at)) - 1],
            },
        )
        for content, start in pieces
    ]


def chunk_by_place(
    lines: list[Line], max_chars: int = 1200, piece_size: int | None = 500
) -> list[Document]:
    """Un documento por lugar; si supera max_chars se divide por sub-viñetas o en trozos"""
    blocks: list[tuple[str, str, list[Line]]] = []
    zone = "General"
    for line in lines:
        if match := _ZONE_RE.match(line.text):
            zone = f"Zona {match.group(1).capitalize()}"
        elif line.text == "OTRAS VAINAS":
            zone = "Otras recomendaciones"
        elif match := _PLACE_RE.match(line.text):
            raw = match.group(1)
            place = "Sitios para comer" if raw.lower().startswith("sitios para comer") else raw
            blocks.append((zone, place.strip(" :"), [line]))
        elif blocks:
            blocks[-1][2].append(line)

    docs: list[Document] = []
    for zone, place, block in blocks:
        is_long = sum(len(line.text) for line in block) > max_chars
        has_subitems = any(_SUBITEM_RE.match(line.text) for line in block)
        if is_long and has_subitems:
            groups: list[tuple[str, list[Line]]] = [("", [])]
            for line in block:
                if match := _SUBITEM_RE.match(line.text):
                    name = re.split(r"\s*[(\[:]", match.group(1), maxsplit=1)[0].strip()
                    groups.append((name, []))
                groups[-1][1].append(line)
            for name, group in groups:
                if group:
                    label = f"{place} – {name}" if name else place
                    docs += _documents(f"{zone} · {label}", group, zone, label, None)
        else:
            size = piece_size if is_long else None
            docs += _documents(f"{zone} · {place}", block, zone, place, size)
    return docs


def chunk_recursive(lines: list[Line], size: int = 500, overlap: int = 100) -> list[Document]:
    """Línea base: RecursiveCharacterTextSplitter de LangChain, sin tener en cuenta la estructura"""
    text = ""
    page_at: list[int] = []
    for line in lines:
        piece = _clean(line.text) + "\n"
        text += piece
        page_at.extend([line.page] * len(piece))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True
    )
    docs = splitter.create_documents([text])
    for doc in docs:
        start = doc.metadata["start_index"]
        doc.metadata = {
            "page_start": page_at[start],
            "page_end": page_at[start + len(doc.page_content) - 1],
        }
    return docs
