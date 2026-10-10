"""Small lexical PDF page retriever with source filename/page citations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pymupdf


POLICY_FILES = (
    "01_Rental_Agreement_and_Terms.pdf",
    "02_Cancellation_and_Refund_Policy_v1.pdf",
    "03_Cancellation_and_Refund_Policy_v2.pdf",
    "04_Late_Return_and_Damage_Penalty_Schedule.pdf",
    "05_EV_User_Guide_and_FAQ.pdf",
)


@dataclass(frozen=True)
class PolicyPage:
    filename: str
    page: int
    text: str

    @property
    def source(self) -> str:
        return f"{self.filename}, page {self.page}"


class PolicyRetrievalError(Exception):
    """Raised when the policy pack cannot be read."""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class PolicyRetriever:
    def __init__(self, root: Path) -> None:
        self.pages: list[PolicyPage] = []
        missing: list[str] = []
        for filename in POLICY_FILES:
            path = root / filename
            if not path.is_file():
                missing.append(filename)
                continue
            try:
                with pymupdf.open(path) as document:
                    for index, page in enumerate(document):
                        text = page.get_text("text").strip()
                        if text:
                            self.pages.append(PolicyPage(filename, index + 1, text))
            except (OSError, RuntimeError, pymupdf.FileDataError) as error:
                raise PolicyRetrievalError(
                    f"Could not read policy document {filename} ({type(error).__name__})."
                ) from error
        if missing:
            raise PolicyRetrievalError(
                "Missing policy document(s): " + ", ".join(missing)
            )
        if not self.pages:
            raise PolicyRetrievalError(
                "No searchable text was extracted from the policy PDFs."
            )

    def search(self, query: str, limit: int = 4) -> list[PolicyPage]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scored: list[tuple[int, PolicyPage]] = []
        for page in self.pages:
            page_tokens = _tokens(page.text)
            score = len(query_tokens & page_tokens)
            if score:
                scored.append((score, page))
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].filename,
                item[1].page,
            )
        )
        return [page for _, page in scored[:limit]]


def format_policy_sources(pages: Iterable[PolicyPage]) -> str:
    return "\n\n".join(
        f"[Source: {page.source}]\n{page.text[:3000]}" for page in pages
    )
