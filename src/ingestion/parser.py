import re
from dataclasses import dataclass, field
from pathlib import Path
import fitz


@dataclass
class ParsedPaper:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    body: str = ""
    source_path: str = ""

    @property
    def full_text(self) -> str:
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        if self.body:
            parts.append(self.body)
        return "\n\n".join(parts)


def parse_pdf(pdf_path: str | Path) -> ParsedPaper:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    paper_id = pdf_path.stem

    pages_text = []
    for page in doc:
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        pages_text.append(text)

    full_text = "\n".join(pages_text)
    metadata = doc.metadata or {}
    title = metadata.get("title", "") or pages_text[0].split("\n")[0][:100] if pages_text else ""
    
    abstract_match = re.search(r"abstract\s*\n(.*?)(?=\n\s*(?:1[\.\s]|introduction))", 
                               full_text, re.IGNORECASE | re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    doc.close()
    return ParsedPaper(
        paper_id=paper_id, title=title.strip(), abstract=abstract,
        body=full_text, source_path=str(pdf_path)
    )


def parse_directory(dir_path: str | Path) -> list[ParsedPaper]:
    dir_path = Path(dir_path)
    papers = []
    for pdf in sorted(dir_path.glob("*.pdf")):
        try:
            papers.append(parse_pdf(pdf))
            print(f"  Parsed: {pdf.name}")
        except Exception as e:
            print(f"  Failed: {pdf.name} - {e}")
    return papers