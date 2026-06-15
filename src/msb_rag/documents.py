from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


DOC_ID_RE = re.compile(r"\*\*Документ №\s*([^*]+?)\*\*")
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s*(.+)$")


@dataclass(frozen=True)
class DocumentChunk:
    """A retrievable fragment of a regulatory markdown document."""

    chunk_id: str
    source: str
    document_id: str
    document_title: str
    section: str
    section_title: str
    heading_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int

    @property
    def anchors(self) -> tuple[str, ...]:
        if not self.section:
            return (self.source,)
        parts = self.section.split(".")
        refs = [f"{self.source}#{'.'.join(parts[:idx])}" for idx in range(1, len(parts) + 1)]
        return tuple(refs)

    @property
    def citation(self) -> str:
        if self.section:
            return f"{self.source}#{self.section}"
        return self.source


@dataclass(frozen=True)
class _SectionBlock:
    source: str
    document_id: str
    document_title: str
    section: str
    section_title: str
    heading_path: tuple[str, ...]
    lines: tuple[str, ...]
    start_line: int
    end_line: int


def load_document_chunks(
    documents_dir: str | Path,
    *,
    target_chars: int = 1_800,
    overlap_paragraphs: int = 1,
) -> list[DocumentChunk]:
    """Load markdown regulations and split them into retrievable chunks."""

    documents_path = Path(documents_dir)
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_path}")

    chunks: list[DocumentChunk] = []
    for path in sorted(documents_path.glob("*.md")):
        blocks = _parse_markdown_sections(path)
        chunks.extend(
            _chunk_blocks(
                _add_outline_blocks(blocks),
                target_chars=target_chars,
                overlap_paragraphs=overlap_paragraphs,
            )
        )
    if not chunks:
        raise ValueError(f"No markdown documents found in {documents_path}")
    return chunks


def _parse_markdown_sections(path: Path) -> list[_SectionBlock]:
    text = _read_utf8(path)
    lines = text.splitlines()
    document_title = _clean_heading(lines[0]) if lines else path.stem
    document_id = _extract_document_id(lines[:20]) or path.stem

    blocks: list[_SectionBlock] = []
    heading_stack: dict[int, str] = {}
    current_meta = {
        "section": "",
        "section_title": document_title,
        "heading_path": (document_title,),
        "start_line": 1,
    }
    current_lines: list[str] = []

    def flush(end_line: int) -> None:
        if not current_lines:
            return
        text_lines = tuple(line.rstrip() for line in current_lines)
        if not "\n".join(text_lines).strip():
            current_lines.clear()
            return
        blocks.append(
            _SectionBlock(
                source=path.name,
                document_id=document_id,
                document_title=document_title,
                section=str(current_meta["section"]),
                section_title=str(current_meta["section_title"]),
                heading_path=tuple(current_meta["heading_path"]),
                lines=text_lines,
                start_line=int(current_meta["start_line"]),
                end_line=end_line,
            )
        )
        current_lines.clear()

    for line_no, line in enumerate(lines, start=1):
        heading = _parse_heading(line)
        if heading:
            level, title = heading
            if level <= 4:
                flush(line_no - 1)
                heading_stack = {k: v for k, v in heading_stack.items() if k < level}
                heading_stack[level] = title
                section, section_title = _parse_section(title)
                path_titles = tuple(heading_stack[k] for k in sorted(heading_stack))
                current_meta = {
                    "section": section,
                    "section_title": section_title,
                    "heading_path": path_titles,
                    "start_line": line_no,
                }
        current_lines.append(line)

    flush(len(lines))
    return blocks


def _chunk_blocks(
    blocks: list[_SectionBlock],
    *,
    target_chars: int,
    overlap_paragraphs: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    counters: dict[str, int] = {}
    for block in blocks:
        paragraphs = _split_paragraphs(block.lines)
        if not paragraphs:
            continue
        current: list[str] = []
        current_start = paragraphs[0][0]

        def emit(end_line: int) -> None:
            if not current:
                return
            body = "\n\n".join(paragraph for _, _, paragraph in current).strip()
            if not body:
                return
            key = f"{block.source}#{block.section or 'root'}"
            counters[key] = counters.get(key, 0) + 1
            suffix = counters[key]
            chunk_id = f"{key}:{suffix}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=block.source,
                    document_id=block.document_id,
                    document_title=block.document_title,
                    section=block.section,
                    section_title=block.section_title,
                    heading_path=block.heading_path,
                    text=_with_context(block, body),
                    start_line=current_start,
                    end_line=end_line,
                )
            )

        for relative_start, relative_end, paragraph in paragraphs:
            start_line = block.start_line + relative_start - 1
            end_line = block.start_line + relative_end - 1
            next_len = len("\n\n".join([p for _, _, p in current] + [paragraph]))
            if current and next_len > target_chars:
                emit(current[-1][1])
                keep = current[-overlap_paragraphs:] if overlap_paragraphs else []
                current = list(keep)
                current_start = current[0][0] if current else start_line
            if not current:
                current_start = start_line
            current.append((start_line, end_line, paragraph))
        emit(current[-1][1])
    return chunks


def _add_outline_blocks(blocks: list[_SectionBlock]) -> list[_SectionBlock]:
    by_parent: dict[tuple[str, str], list[_SectionBlock]] = {}
    indexed = {block.section: block for block in blocks if block.section}
    for block in blocks:
        if not block.section or "." not in block.section:
            continue
        parent = block.section.rsplit(".", 1)[0]
        if parent in indexed:
            by_parent.setdefault((block.source, parent), []).append(block)

    outlines: list[_SectionBlock] = []
    for (source, parent), children in by_parent.items():
        parent_block = indexed[parent]
        direct_children = [
            child
            for child in children
            if child.section.count(".") == parent.count(".") + 1
        ]
        if len(direct_children) < 2:
            continue
        lines = [
            f"{'#' * len(parent_block.heading_path)} {parent_block.section}. {parent_block.section_title}",
            "",
            "Подразделы:",
        ]
        for child in direct_children:
            lines.append(f"- {child.section}. {child.section_title}")
        outlines.append(
            _SectionBlock(
                source=source,
                document_id=parent_block.document_id,
                document_title=parent_block.document_title,
                section=parent_block.section,
                section_title=parent_block.section_title,
                heading_path=parent_block.heading_path,
                lines=tuple(lines),
                start_line=parent_block.start_line,
                end_line=parent_block.end_line,
            )
        )
    return outlines + blocks


def _split_paragraphs(lines: tuple[str, ...]) -> list[tuple[int, int, str]]:
    paragraphs: list[tuple[int, int, str]] = []
    current: list[str] = []
    start_line = 1
    for idx, line in enumerate(lines, start=1):
        if line.strip():
            if not current:
                start_line = idx
            current.append(line.rstrip())
            continue
        if current:
            paragraphs.append((start_line, idx - 1, "\n".join(current).strip()))
            current = []
    if current:
        paragraphs.append((start_line, len(lines), "\n".join(current).strip()))
    return paragraphs


def _with_context(block: _SectionBlock, body: str) -> str:
    heading = " > ".join(block.heading_path)
    return (
        f"Источник: {block.source}\n"
        f"Документ: {block.document_id}\n"
        f"Раздел: {block.section or 'без номера'} {block.section_title}\n"
        f"Путь: {heading}\n\n"
        f"{body}"
    ).strip()


def _clean_heading(line: str) -> str:
    heading = _parse_heading(line)
    return heading[1] if heading else line.strip()


def _read_utf8(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _parse_heading(line: str) -> tuple[int, str] | None:
    if not line.startswith("#"):
        return None
    level = 0
    for char in line[:6]:
        if char != "#":
            break
        level += 1
    if level == 0 or len(line) <= level or line[level] != " ":
        return None
    return level, line[level + 1 :].strip()


def _parse_section(title: str) -> tuple[str, str]:
    match = SECTION_RE.match(title)
    if match:
        return match.group(1), match.group(2)
    if title.startswith("Приложение ") and "." in title:
        section, section_title = title.split(".", 1)
        return section.strip(), section_title.strip()
    if title.startswith("Приложение "):
        return title.strip(), title.strip()
    return "", title


def _extract_document_id(lines: list[str]) -> str:
    for line in lines:
        match = DOC_ID_RE.search(line)
        if match:
            return match.group(1).strip()
    return ""
