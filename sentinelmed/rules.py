"""Step 1 - Detect: regular expressions for identifiers with a fixed Mexican format.

Where the format carries a check digit (CURP, NSS) it is verified, because a
valid check digit is strong evidence and a failed one is a reason to ask a
person rather than to ignore the match.
"""
from __future__ import annotations

import re

from .lexicons import NSS_TRIGGERS, NUMBER_PAPERWORK, PHONE_TRIGGERS
from .textutil import norm
from .types import Finding

# ---------------------------------------------------------------- CURP
CURP_STATES = ("AS BC BS CC CL CM CS CH DF DG GT GR HG JC MC MN MS NT NL OC PL QT QR SP SL SR TC TS "
               "TL VZ YN ZS NE").split()
_DATE6 = r"\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
CURP_RE = re.compile(
    rf"(?<![A-Za-z0-9])[A-Z][AEIOUX][A-Z]{{2}}[ -]?{_DATE6}[ -]?[HMX][ -]?(?:{'|'.join(CURP_STATES)})"
    rf"[ -]?[B-DF-HJ-NP-TV-Z]{{3}}[ -]?[A-Z0-9]\d(?![A-Za-z0-9])",
    re.I,
)
_CURP_ALPHABET = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"


def curp_check_digit(first17: str) -> str:
    total = sum(_CURP_ALPHABET.index(ch) * (18 - i) for i, ch in enumerate(first17.upper()))
    return str((10 - total % 10) % 10)


def curp_is_valid(curp: str) -> bool:
    c = re.sub(r"[ -]", "", curp).upper()
    return len(c) == 18 and curp_check_digit(c[:17]) == c[17]


# ---------------------------------------------------------------- NSS (IMSS)
NSS_RE = re.compile(r"(?<![\w.-])\d{2}[ -]?\d{2}[ -]?\d{2}[ -]?\d{4}[ -]?\d(?![\w-])")


def luhn_check_digit(digits: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def nss_is_valid(nss: str) -> bool:
    d = re.sub(r"\D", "", nss)
    return len(d) == 11 and luhn_check_digit(d[:10]) == d[10]


# ---------------------------------------------------------------- RFC, e-mail, phone, birth date
RFC_RE = re.compile(rf"(?<![A-Za-z0-9])[A-ZÑ&]{{4}}[ -]?{_DATE6}[ -]?[A-Z0-9]{{3}}(?![A-Za-z0-9])", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A run of digit groups: "686 123 4567", "(664) 811-0240", "664 1 97 90 67", "+52 55 1234 5678".
PHONE_RE = re.compile(r"(?<![\w/.,:-])\+?\(?\d+\)?(?:(?:[ .-]{1,2}|(?<=\)))\(?\d+\)?)*(?![\w/-])(?!\.\d)")
_PHONE_EXT_RE = re.compile(r"\s*,?\s*(?:ext\.?|extensi[oó]n)\s*\d{1,5}", re.I)
_PHONE_PREFIX = re.compile(r"^(?:521|52|01|044|045)(?=\d{10}$)")
_MONTHS = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
_DATE = (rf"(?:\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}"
         rf"|\d{{1,2}}\s+de\s+(?:{_MONTHS})\s+(?:de|del)\s+\d{{4}})")
# A date shortly after a word about birth: "Fecha nac. 08/01/98", "Nacida el 3 de mayo de 1990".
BIRTHDATE_RE = re.compile(
    rf"(?:fecha\s+(?:de\s+)?nac(?:imiento)?|\bf\.?\s?nac|\bf\.?\s?n\b|nacid[oa]|naci[oó]|nacimiento|\bdob)\.?"
    rf"[^\d\n;]{{0,25}}?({_DATE})",
    re.I,
)


def _words_before(text: str, start: int, n: int = 4) -> list[str]:
    return [norm(w) for w in re.findall(r"[^\W\d_]+", text[max(0, start - 45):start])[-n:]]


def _is_phone(digits: str) -> bool:
    return len(_PHONE_PREFIX.sub("", digits)) == 10


def _phone_windows(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """The part(s) of a digit run that make a 10-digit Mexican number."""
    run = text[start:end]
    digits = re.sub(r"\D", "", run)
    if _is_phone(digits):
        return [(start, end)]
    if len(digits) <= 11:                 # too short, or NSS-shaped: not a phone
        return []
    groups = [(g.start(), g.end()) for g in re.finditer(r"\+?\(?\d+\)?", run)]
    out, i = [], 0
    while i < len(groups):                # a longer run: look for a phone inside it
        for j in range(i, len(groups)):
            d = re.sub(r"\D", "", run[groups[i][0]:groups[j][1]])
            if _is_phone(d):
                out.append((start + groups[i][0], start + groups[j][1]))
                i = j
                break
            if len(d) > 13:
                break
        i += 1
    return out


def find_structured(text: str) -> list[Finding]:
    """All rule-based candidates in ``text``. Overlaps are resolved later."""
    out: list[Finding] = []

    for m in CURP_RE.finditer(text):
        ok = curp_is_valid(m.group())
        out.append(Finding(m.start(), m.end(), m.group(), "CURP", {"rule"}, 0.99 if ok else 0.85, ok))

    for m in RFC_RE.finditer(text):
        out.append(Finding(m.start(), m.end(), m.group(), "RFC", {"rule"}, 0.9))

    for m in EMAIL_RE.finditer(text):
        out.append(Finding(m.start(), m.end(), m.group(), "EMAIL", {"rule"}, 0.99))

    for m in BIRTHDATE_RE.finditer(text):
        out.append(Finding(m.start(1), m.end(1), m.group(1), "BIRTHDATE", {"rule"}, 0.95))

    for m in NSS_RE.finditer(text):
        ok = nss_is_valid(m.group())
        before = set(_words_before(text, m.start()))
        if before & NSS_TRIGGERS:
            prior = 0.97 if ok else 0.8
        elif before & NUMBER_PAPERWORK:
            prior = 0.3 if ok else 0.1
        else:
            prior = 0.6 if ok else 0.25
        out.append(Finding(m.start(), m.end(), m.group(), "NSS", {"rule"}, prior, ok))

    for m in PHONE_RE.finditer(text):
        for start, end in _phone_windows(text, m.start(), m.end()):
            ext = _PHONE_EXT_RE.match(text, end)
            end = ext.end() if ext else end
            chunk = text[start:end]
            before = set(_words_before(text, start))
            if before & PHONE_TRIGGERS:
                prior = 0.97
            elif before & NUMBER_PAPERWORK:
                prior = 0.2
            else:
                prior = 0.9 if re.search(r"[ .()-]", chunk) else 0.6
            out.append(Finding(start, end, chunk, "PHONE", {"rule"}, prior))

    return out
