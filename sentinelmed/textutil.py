"""Small text helpers shared by every layer."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_L = r"[^\W\d_]"          # any Unicode letter (Behçet, Sjögren, Lasègue)
TOKEN_RE = re.compile(rf"{_L}+(?:[-']{_L}+)*|\d+(?:[.,/]\d+)*|\S")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def deaccent(s: str) -> str:
    """Remove accents. Length-preserving for NFC input (ñ becomes n)."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    """Case- and accent-insensitive key used for every lexicon lookup."""
    return deaccent(s).lower()


@dataclass(frozen=True)
class Tok:
    text: str
    start: int
    end: int
    norm: str

    @property
    def is_alpha(self) -> bool:
        return self.text[0].isalpha()

    @property
    def is_cap(self) -> bool:
        return self.text[0].isupper()


def tokenize(text: str) -> list[Tok]:
    return [Tok(m.group(), m.start(), m.end(), norm(m.group())) for m in TOKEN_RE.finditer(text)]


def case_mode(text: str) -> str:
    """'upper' for ALL-CAPS notes, 'lower' for notes typed without capitals, else 'mixed'.

    Capitalisation is the cheapest clue for a name, so each layer needs to know
    when that clue is missing.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 8:
        return "mixed"
    up = sum(c.isupper() for c in letters) / len(letters)
    if up > 0.75:
        return "upper"
    if up < 0.012:
        return "lower"
    return "mixed"
