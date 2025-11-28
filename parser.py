"""Resume parser utilities for PDF/DOCX/TXT files."""
import argparse
from pathlib import Path
from typing import List

from docx import Document
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}


def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    chunks: List[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - defensive log only
            chunks.append(f"\n[page extraction error: {exc}]\n")
    return "\n".join(chunks)


def read_docx(path: Path) -> str:
    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    return read_txt(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from resume files")
    parser.add_argument("input", type=Path, help="Path to PDF/DOCX/TXT file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    text = extract_text(args.input)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
