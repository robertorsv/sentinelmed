"""Step 2 - Understand: names, family references and addresses in free text.

Three sources of evidence are combined, and each candidate remembers which
ones fired so the classifier can weigh them:

* context     - a word that announces a name ("paciente", "Dra.", "su esposa")
* gazetteer   - a small list of common Mexican first names and surnames
* ner         - an optional Spanish spaCy model (the language model in the sheet)
* propagated  - a name already known from a structured column of the same row
"""
from __future__ import annotations

import re
from bisect import bisect_left

from . import lexicons as L
from .textutil import Tok, norm
from .types import Finding

# ---------------------------------------------------------------- optional spaCy layer


class SpacyNER:
    """Thin wrapper so the rest of the code never imports spaCy directly."""

    def __init__(self, model: str = "es_core_news_sm"):
        import spacy  # optional dependency

        self.nlp = spacy.load(model, disable=["morphologizer", "parser", "attribute_ruler", "lemmatizer"])
        self.model = model

    def person_spans(self, texts: list[str]) -> list[list[tuple[int, int]]]:
        return [[(e.start_char, e.end_char) for e in doc.ents if e.label_ == "PER"]
                for doc in self.nlp.pipe(texts, batch_size=64)]


def load_ner(model: str = "es_core_news_sm"):
    """Return a SpacyNER, or None when spaCy or the model is not installed."""
    try:
        return SpacyNER(model)
    except Exception:
        return None


# ---------------------------------------------------------------- names

_GLUE = L.NAME_PARTICLES | {"su", "nombre", "el", "llamado", "llamada"} | L.PATIENT_TRIGGERS
_PATIENT_WORDS = L.PATIENT_TRIGGERS - {"nombre"}
_TITLES = L.DOCTOR_TITLES | L.PERSON_TITLES
_PLACE_WORDS = frozenset(w for place in (L.CITIES | L.STATES) for w in place.replace(".", " ").split())


_INNER_CAP_OK = ("Mc", "Mac", "De", "Di", "La", "Le", "O'", "D'")


def _is_name_word(t: Tok, mode: str) -> bool:
    w = t.norm
    if not t.is_alpha or len(w) < 2 or w in L.STOPWORDS or (mode == "mixed" and not t.is_cap):
        return False
    in_list = w in L.FIRST_NAMES or w in L.SURNAMES
    # "SatO2", "HbA1c": capitals inside a word are lab shorthand, not names
    if not t.text.isupper() and any(c.isupper() for c in t.text[1:]) and not t.text.startswith(_INNER_CAP_OK):
        return False
    # "refiriendo", "caminando": gerunds are not names (Fernando and Armando are in the list)
    if len(w) > 6 and w.endswith(("ando", "iendo", "yendo")) and not in_list:
        return False
    return True


def _role(toks: list[Tok], i0: int, i1: int) -> str | None:
    """Who the name belongs to, read from the words around it."""
    saw_patient = False
    k, steps = i0 - 1, 0
    while k >= 0 and steps < 8:
        t = toks[k]
        if t.text == ".":
            if not (k > 0 and toks[k - 1].norm in _TITLES):
                break
        elif t.is_alpha:
            w = t.norm
            if w in L.DOCTOR_TITLES:
                return "doctor"
            if w in L.KIN_TERMS:
                return "relative"
            if w in _PATIENT_WORDS:
                saw_patient = True
            elif w not in _GLUE and w not in L.PERSON_TITLES:
                break
        elif t.text not in {",", ":", "("}:
            break
        k -= 1
        steps += 1
    # "María López (esposa)", "Ana Ruiz, hija del paciente"
    k = i1 + 1
    while k < len(toks) and k <= i1 + 3 and (toks[k].text in {"(", ","} or toks[k].norm == "su"):
        k += 1
    if k < len(toks) and k > i1 + 1 and toks[k].norm in L.KIN_TERMS:
        return "relative"
    return "patient" if saw_patient else None


def _base_runs(toks: list[Tok], marks: list[set]) -> list[tuple[int, int]]:
    """Join marked tokens into runs, bridging particles ("María de los Ángeles")."""
    n, runs, i = len(toks), [], 0
    while i < n:
        if not marks[i]:
            i += 1
            continue
        j = i
        while True:
            k = j + 1
            if k < n and marks[k]:
                j = k
                continue
            look = k
            while look < n and toks[look].norm in L.NAME_PARTICLES and look - k < 3:
                look += 1
            if look > k and look < n and marks[look]:
                j = look
                continue
            break
        runs.append((i, j))
        i = j + 1
    return runs


def find_names(text: str, toks: list[Tok], mode: str, per_spans=(), known_tokens=frozenset(),
               use_context: bool = True) -> list[Finding]:
    n = len(toks)
    marks: list[set] = [set() for _ in range(n)]
    starts = [t.start for t in toks]

    def ok(k: int) -> bool:
        # A word directly followed by a colon is a field label ("Diagnóstico:"), not a name.
        return _is_name_word(toks[k], mode) and not (k + 1 < n and toks[k + 1].text == ":")

    def back_chain(k: int, limit: int = 4) -> None:
        """Mark the name words that end at token k, walking backwards."""
        count = 0
        while k >= 0 and count < limit:
            if ok(k):
                marks[k].add("context")
                k -= 1
                count += 1
                continue
            look = k
            while look >= 0 and toks[look].norm in L.NAME_PARTICLES and k - look < 3:
                look -= 1
            if look < k and look >= 0 and count > 0 and ok(look):
                k = look
                continue
            break

    # language model
    for s, e in per_spans:
        for k in range(bisect_left(starts, s), n):
            if toks[k].start >= e:
                break
            if ok(k) or (toks[k].is_alpha and toks[k].norm not in L.STOPWORDS and len(toks[k].norm) >= 2
                         and not (k + 1 < n and toks[k + 1].text == ":")):
                marks[k].add("ner")

    for i, t in enumerate(toks):
        if not t.is_alpha or len(t.norm) < 2 or t.norm in L.STOPWORDS or (i + 1 < n and toks[i + 1].text == ":"):
            continue
        w = t.norm
        if use_context and (w in L.FIRST_NAMES or w in L.SURNAMES) and (mode != "mixed" or t.is_cap):
            marks[i].add("gazetteer")
        if w in known_tokens and (t.is_cap or mode != "mixed") and (t.is_cap or w not in L.AMBIGUOUS):
            marks[i].add("propagated")

    if use_context:
        limit = 5 if mode == "mixed" else 4

        def fwd_chain(k: int, after_title: bool) -> None:
            """Mark the name words that start at token k."""
            count = 0
            while k < n and count < limit:
                if toks[k].norm in L.NAME_PARTICLES:
                    look = k
                    while look < n and toks[look].norm in L.NAME_PARTICLES:
                        look += 1
                    if look < n and look - k <= 3 and ok(look):
                        # "Dr. Del Río", "Sra. De la Cruz": the particle opens the surname
                        if count == 0 and (toks[k].is_cap if mode == "mixed" else after_title):
                            for q in range(k, look):
                                marks[q].add("context")
                        k = look
                        continue
                    break
                if not ok(k):
                    break
                marks[k].add("context")
                k += 1
                count += 1

        for i, t in enumerate(toks):
            if not t.is_alpha:
                continue
            w = t.norm
            # 1. a word that announces a name: "paciente", "Dra.", "su esposa", "Contacto:"
            if w in L.NAME_TRIGGERS or w in L.PERSON_LABELS:
                k = i + 1
                skip = {":", ",", "(", "-"} | ({"."} if w in L.ABBREVIATIONS else set())
                while k < n and toks[k].text in skip:
                    k += 1
                fwd_chain(k, w in _TITLES)
            # ... or a verb that takes a person: "Ingresa X", "se valora a X", "se trata de X"
            if w in L.PERSON_VERBS:
                fwd_chain(i + 1, False)
            if w in L.OBJECT_VERBS and i + 2 < n and toks[i + 1].norm in {"a", "de", "al"}:
                fwd_chain(i + 2, False)
            # 2. the subject of a verb only people do: "Itzel refiere", "KEVIN OLACHEA ACUDE"
            if w in L.PERSON_VERBS:
                back_chain(i - 1)
            # 3. apposition with a family word: "Efraín, su hijo," and "Ana Ruiz (hija)"
            if w in L.KIN_TERMS:
                if i >= 3 and toks[i - 1].norm == "su" and toks[i - 2].text == ",":
                    back_chain(i - 3)
                elif i >= 2 and toks[i - 1].text == "(":
                    back_chain(i - 2)

        # 4. shape: in normally-capitalised Spanish, a capital in the middle of a
        #    sentence is a proper noun, and a run of ALL-CAPS words is usually a name
        #    written surname-first.
        if mode == "mixed":
            def plain(k: int) -> bool:
                w = toks[k].norm
                return (ok(k) and len(w) >= 3 and w not in L.DRUGS and w not in L.MEDICAL_TERMS
                        and w not in L.INSTITUTION_WORDS and w not in _PLACE_WORDS)
            for i, t in enumerate(toks):
                if not (t.is_alpha and plain(i)):
                    continue
                if t.text.isupper():
                    # neighbours, also across a comma: "CUÉLLAR BALDERAS, CASIMIRO"
                    cand = [i - 1, i + 1]
                    if i >= 2 and toks[i - 1].text == ",":
                        cand.append(i - 2)
                    if i + 2 < n and toks[i + 1].text == ",":
                        cand.append(i + 2)
                    near = [k for k in cand if 0 <= k < n and toks[k].is_alpha and toks[k].text.isupper()
                            and plain(k)]
                    if near:
                        marks[i].add("shape")
                elif i > 0 and toks[i - 1].text not in {".", ":", ";", "?", "!", "-", "•"}:
                    marks[i].add("shape")

    # A capitalised (or, without capitals, any unknown) neighbour is probably the
    # rest of the name: an unseen second surname, or a given name not in the list.
    def extendable(k: int) -> bool:
        w = toks[k].norm
        return (ok(k) and not marks[k] and w not in L.DRUGS and w not in L.MEDICAL_TERMS
                and w not in L.INSTITUTION_WORDS)

    for i0, i1 in _base_runs(toks, marks):
        srcs = set().union(*marks[i0:i1 + 1])
        single = i0 == i1
        if mode != "mixed" and single and toks[i0].norm in L.AMBIGUOUS and not (srcs & {"context", "ner", "propagated"}):
            continue
        j = i1
        for _ in range(2):
            k = j + 1
            while k < n and k - j <= 3 and toks[k].norm in L.NAME_PARTICLES:
                k += 1
            if k < n and extendable(k):
                marks[k].add("extend")
                j = k
            else:
                break
        j = i0
        for _ in range(2):
            if j - 1 >= 0 and extendable(j - 1):
                marks[j - 1].add("extend")
                j -= 1
            else:
                break

    out: list[Finding] = []
    for i0, i1 in _base_runs(toks, marks):
        sources = set().union(*marks[i0:i1 + 1])
        n_alpha = sum(1 for t in toks[i0:i1 + 1] if t.norm not in L.NAME_PARTICLES)
        if ("propagated" in sources or ("context" in sources and "gazetteer" in sources)
                or ("ner" in sources and "gazetteer" in sources)):
            prior = 0.95
        elif "gazetteer" in sources and n_alpha >= 2:
            prior = 0.9
        elif "context" in sources:
            prior = 0.7
        elif "ner" in sources:
            prior = 0.6
        elif "gazetteer" in sources:
            prior = 0.5
        else:
            prior = 0.35
        s, e = toks[i0].start, toks[i1].end
        out.append(Finding(s, e, text[s:e], "NAME", sources, prior, role=_role(toks, i0, i1)))
    return out


# ---------------------------------------------------------------- addresses

_NUM_RE = re.compile(r"\s*,?\s*(?:(?:#|n[o°º]\.?|num\.?|numero)\s*)?\d{1,5}(?:\s?-\s?[a-z0-9]{1,3}\b)?(?![\d/])|\s+s/n\b")
_INT_RE = re.compile(r"\s*,?\s*(?:int\.?|interior|depto\.?|dpto\.?|departamento|local|lote|manzana|mz\.?|lt\.?"
                     r"|edificio|edif\.?)\s*#?\s*[a-z0-9]{1,4}\b")
_COL_RE = re.compile(r"\s*,?\s*(?:col\.?|colonia|fracc\.?|fraccionamiento|ejido|barrio|residencial|infonavit)\s+")
_CP_RE = re.compile(r"\s*,?\s*(?:c\.?\s?p\.?|codigo postal)\s*:?\s*\d{5}\b")
# lot-and-block addresses: "Mz. 48 Lt. 22", "Lote 5 Manzana 12"
_BLOCK_RE = re.compile(r"(?:mz\.?|mza\.?|manzana)\s*\d+[a-z]?\s*,?\s*(?:lt\.?|lote)\s*\d+[a-z]?"
                       r"|(?:lt\.?|lote)\s*\d+[a-z]?\s*,?\s*(?:mz\.?|mza\.?|manzana)\s*\d+[a-z]?")
_PLACE_RE = re.compile(
    r"\s*,\s*(?:" + "|".join(re.escape(p) for p in sorted(L.CITIES | L.STATES, key=len, reverse=True)) + r")(?![a-z])"
)
_ADDR_PARTICLES = L.NAME_PARTICLES | {"el", "y", "a"}
_ADDR_BREAK = {"no", "num", "numero", "int", "interior", "cp", "tel", "cel", "telefono", "celular", "paciente", "px",
               "curp", "nss", "correo", "entre", "esq", "esquina"}
_ADDR_COMMON = {"centro", "nacional", "social", "salud", "general", "unidad", "principal", "hospital", "seguro"}


def _addr_word_ok(w: str) -> bool:
    return w in _ADDR_COMMON or w not in L.STOPWORDS


def _name_words(toks: list[Tok], j: int, mode: str, maxn: int) -> int:
    """Consume the words of a street or colonia name starting at token j. Returns the end index."""
    n, count = len(toks), 0
    while j < n and count < maxn:
        t = toks[j]
        if t.is_alpha:
            w = t.norm
            if w in L.COLONIA_WORDS or w in _ADDR_BREAK:
                break
            if w == "c" and j + 1 < n and (toks[j + 1].text == "." or toks[j + 1].norm == "p"):   # "C.P."
                break
            if w in _ADDR_PARTICLES:
                look = j
                while look < n and toks[look].norm in _ADDR_PARTICLES:
                    look += 1
                ok = look < n and look - j <= 3 and toks[look].is_alpha and (
                    toks[look].is_cap if mode == "mixed" else _addr_word_ok(toks[look].norm))
                if not ok:
                    break
                count += look - j
                j = look
                continue
            # With capitals, a capitalised word is enough ("Col. Centro"). Without them,
            # stop at the first ordinary word.
            if (mode == "mixed" and not t.is_cap) or (mode != "mixed" and not _addr_word_ok(w)):
                break
        elif t.text.isdigit():
            nxt = toks[j + 1] if j + 1 < n else None
            date_named = nxt is not None and nxt.norm == "de" and j + 2 < n and toks[j + 2].is_alpha  # "5 de Mayo"
            if not (date_named or (count == 0 and len(t.text) <= 3)):
                break
            if date_named:
                j += 3
                count += 3
                continue
        else:
            break
        j += 1
        count += 1
    return j


def find_addresses(text: str, toks: list[Tok], mode: str) -> list[Finding]:
    ntext = norm(text)
    starts = [t.start for t in toks]
    n = len(toks)
    out: list[Finding] = []
    i = 0
    while i < n:
        t = toks[i]
        kind = None
        block = _BLOCK_RE.match(ntext, t.start) if t.norm in {"mz", "mza", "manzana", "lt", "lote"} else None
        if block:
            kind = "block"
        elif t.is_alpha and t.norm in L.STREET_TYPES:
            kind = "street"
        elif t.norm == "c" and i + 1 < n and toks[i + 1].text == "." and mode != "lower":   # "C. Hidalgo 12"
            kind = "street"
        elif t.is_alpha and t.norm in L.COLONIA_WORDS:
            kind = "colonia"
        elif t.is_alpha and t.norm in L.ADDRESS_TRIGGERS:
            kind = "trigger"
        if kind is None:
            i += 1
            continue

        j = i + 1
        if kind == "trigger":
            while j < n and (toks[j].text in {":", ","}
                             or toks[j].norm in {"en", "por", "sobre", "la", "el", "actual", "particular"}):
                j += 1
            if j < n and (toks[j].norm in L.STREET_TYPES or toks[j].norm in L.COLONIA_WORDS):
                i = j          # the street/colonia branch will handle it
                continue
            first = j
        else:
            if j < n and toks[j].text == ".":
                j += 1
            first = i
        has_colonia = kind == "colonia"
        has_number = False
        if kind == "block":
            pos, has_number = block.end(), True
        else:
            j2 = _name_words(toks, j, mode, 6)
            if j2 == j:
                i += 1
                continue
            pos = toks[j2 - 1].end
            m = _NUM_RE.match(ntext, pos)
            if m:
                pos, has_number = m.end(), True
        progressed = True
        while progressed:
            progressed = False
            for rx in (_INT_RE, _CP_RE, _PLACE_RE):
                m = rx.match(ntext, pos)
                if m:
                    pos, progressed = m.end(), True
            m = _COL_RE.match(ntext, pos)
            if m:
                cj = bisect_left(starts, m.end())
                cj2 = _name_words(toks, cj, mode, 5)
                if cj2 > cj:
                    pos, progressed, has_colonia = toks[cj2 - 1].end, True, True

        ambiguous_type = kind == "street" and (t.norm in L.AMBIGUOUS_STREET_TYPES or t.norm == "c")
        needs_support = kind == "trigger" or ambiguous_type or (mode != "mixed" and kind == "street")
        if needs_support and not (has_number or has_colonia):
            i += 1
            continue
        if kind == "trigger" and not has_number:
            i += 1
            continue

        s = toks[first].start
        prior = 0.95 if has_number else (0.75 if kind == "street" else 0.7)
        out.append(Finding(s, pos, text[s:pos], "ADDRESS", {"rule", "context"} if kind == "trigger" else {"rule"}, prior))
        i = max(bisect_left(starts, pos), i + 1)
    return out
