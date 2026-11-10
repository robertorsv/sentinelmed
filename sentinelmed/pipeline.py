"""Detect -> Understand -> Protect, for one text or for a whole table."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import lexicons as L
from .classifier import DEFAULT_MODEL, Confirmer, featurize, rejection_reason
from .language import find_addresses, find_names, load_ner
from .rules import find_structured
from .textutil import case_mode, nfc, norm, tokenize
from .types import PRIORITY, Finding

# ---------------------------------------------------------------- Step 3 - Protect


class Vault:
    """Hands out consistent placeholders: the same value always gets the same tag."""

    def __init__(self):
        self._ids: dict[str, dict[str, int]] = {}
        self._originals: dict[str, str] = {}

    def placeholder(self, category: str, key: str, original: str = "") -> str:
        ids = self._ids.setdefault(category, {})
        if key not in ids:
            ids[key] = len(ids) + 1
        tag = f"[{category}_{ids[key]}]"
        self._originals.setdefault(tag, original)
        return tag

    def key_table(self) -> pd.DataFrame:
        """Placeholder -> original value. This is the re-identification key: keep it apart or do not write it."""
        return pd.DataFrame(sorted(self._originals.items()), columns=["placeholder", "original"])


def _name_tokens(s: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[^\W\d_]+", norm(s)) if w not in L.NAME_PARTICLES and len(w) >= 2)


class _People:
    """Who is who inside one record, so 'Juan Pérez' and 'Sr. Pérez' share a placeholder."""

    def __init__(self, record_key: str, patient_tokens: frozenset[str] = frozenset()):
        self.record_key = record_key
        self.patient_tokens = patient_tokens
        self.seen: list[tuple[str, frozenset[str], str]] = []

    def resolve(self, f: Finding) -> tuple[str, str]:
        toks = _name_tokens(f.text)
        if f.role == "doctor":
            category = "DOCTOR"
        elif f.role == "relative":
            category = "RELATIVE"
        elif self.patient_tokens and toks and toks <= self.patient_tokens:
            category = "PATIENT"
        elif f.role == "patient" and not self.patient_tokens:
            self.patient_tokens, category = toks, "PATIENT"
        else:
            category = "PERSON"
        if category == "PATIENT":
            return category, " ".join(sorted(self.patient_tokens))
        for cat, old, key in self.seen:
            if cat == category and toks and (toks <= old or old <= toks):
                return category, key
        scope = "" if category == "DOCTOR" else f"{self.record_key}|"
        key = scope + " ".join(sorted(toks))
        self.seen.append((category, toks, key))
        return category, key


_MONTH_NUM = {m: i + 1 for i, m in enumerate(
    "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split())}


def _date_key(text: str) -> str:
    """One key for the same birth date written in different ways (15/03/1972, 1972-03-15, 15 de marzo de 1972)."""
    t = norm(text)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{d:02d}-{mo:02d}-{y % 100:02d}"
    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", t)
    if m:
        d, mo, y = map(int, m.groups())
        return f"{d:02d}-{mo:02d}-{y % 100:02d}"
    m = re.fullmatch(r"(\d{1,2})\s+de\s+([a-z]+)\s+del?\s+(\d{4})", t)
    if m and m.group(2) in _MONTH_NUM:
        return f"{int(m.group(1)):02d}-{_MONTH_NUM[m.group(2)]:02d}-{int(m.group(3)) % 100:02d}"
    return t


def _value_key(label: str, text: str) -> str:
    if label == "BIRTHDATE":
        return _date_key(text.strip())
    if label in ("PHONE", "NSS"):
        d = re.sub(r"\D", "", text)
        return d[-10:] if label == "PHONE" else d
    if label in ("CURP", "RFC"):
        return re.sub(r"[ -]", "", text).upper()
    return re.sub(r"\s+", " ", norm(text)).strip(" .,")


def apply_placeholders(text: str, findings: list[Finding], vault: Vault, people: _People) -> str:
    """Replace every masked finding. Longest patient name first so shorter mentions can link to it."""
    masked = [f for f in findings if f.masked]
    for f in sorted(masked, key=lambda f: -(f.end - f.start)):
        if f.label == "NAME":
            category, key = people.resolve(f)
        else:
            category, key = f.label, _value_key(f.label, f.text)
        f.placeholder = vault.placeholder(category, key, f.text)
    out, pos = [], 0
    for f in sorted(masked, key=lambda f: f.start):
        out.append(text[pos:f.start])
        out.append(f.placeholder)
        pos = f.end
    out.append(text[pos:])
    return "".join(out)


# ---------------------------------------------------------------- the pipeline


def _resolve_overlaps(cands: list[Finding]) -> list[Finding]:
    """Keep the most specific candidate where two overlap. Rejected candidates never block a masked one."""
    kept: list[Finding] = []

    def free(f):
        return all(f.end <= k.start or f.start >= k.end for k in kept)

    order = sorted(cands, key=lambda f: (not f.masked, f.decision == "skip", PRIORITY[f.label], -(f.end - f.start)))
    for f in order:
        if free(f):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)


class SentinelMed:
    def __init__(self, use_context: bool = True, use_ner: bool = True, use_classifier: bool = True,
                 spacy_model: str = "es_core_news_sm", model_path: str | Path | None = None,
                 denylist: list[str] | None = None, ner=None, confirmer: Confirmer | None = None):
        self.use_context = use_context
        self.ner = ner if ner is not None else (load_ner(spacy_model) if use_ner else None)
        self.confirmer = confirmer
        if use_classifier and confirmer is None:
            path = Path(model_path) if model_path else DEFAULT_MODEL
            self.confirmer = Confirmer.load(path) if path.exists() else None
        self.denylist = [d for d in (denylist or []) if d.strip()]

    # -- candidates ---------------------------------------------------------
    def candidates(self, text: str, per_spans=(), known_tokens=frozenset()):
        toks = tokenize(text)
        mode = case_mode(text)
        cands = find_structured(text)
        if self.use_context:
            cands += find_addresses(text, toks, mode)
        if self.use_context or per_spans or known_tokens:
            cands += find_names(text, toks, mode, per_spans, known_tokens, self.use_context)
        for term in self.denylist:
            for m in re.finditer(re.escape(term), text, re.I):
                cands.append(Finding(m.start(), m.end(), m.group(), "CUSTOM", {"custom"}, 1.0))
        starts = [t.start for t in toks]
        for f in cands:
            f.features = featurize(text, toks, starts, f, mode)
        return cands

    # -- full analysis ------------------------------------------------------
    def analyze(self, texts: list[str], known: list[frozenset] | None = None) -> list[list[Finding]]:
        texts = [nfc(t) for t in texts]
        known = known or [frozenset()] * len(texts)
        spans = self.ner.person_spans(texts) if self.ner else [()] * len(texts)
        all_cands = [self.candidates(t, s, k) for t, s, k in zip(texts, spans, known)]
        flat = [f for cands in all_cands for f in cands]
        if self.confirmer:
            for f, p in zip(flat, self.confirmer.score([f.features for f in flat])):
                if f.sources & {"custom", "column"} or f.label == "EMAIL" or (f.label == "CURP" and f.valid):
                    p = max(p, 0.99)
                f.confidence, f.decision = p, self.confirmer.decide(p)
                if not f.masked:
                    f.reason = rejection_reason(f) or ""
                    if not f.reason:            # recall first: no concrete reason, no rejection
                        f.decision = "review"
        else:
            for f in flat:                      # no second reviewer: every candidate is masked
                f.confidence, f.decision = f.prior, "mask"
        return [_resolve_overlaps(c) for c in all_cands]

    def mask_text(self, text: str, vault: Vault | None = None, record_key: str = "0"):
        text = nfc(text)
        findings = self.analyze([text])[0]
        return apply_placeholders(text, findings, vault or Vault(), _People(record_key)), findings

    # -- tables -------------------------------------------------------------
    def mask_table(self, df: pd.DataFrame, text_cols: list[str] | None = None,
                   id_cols: dict[str, str] | None = None):
        """Returns (masked DataFrame, findings table, vault)."""
        id_cols = id_cols if id_cols is not None else guess_id_columns(df)
        text_cols = text_cols if text_cols is not None else guess_text_columns(df, id_cols)
        other_cols = [c for c in df.columns if c not in id_cols and c not in text_cols
                      and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]
        out = df.copy()
        for c in list(id_cols) + text_cols + other_cols:
            out[c] = out[c].astype(object)
        vault, rows = Vault(), []
        keys = [str(i) for i in df.index]

        patient_tokens = []
        for i in df.index:
            toks = set()
            for c, lab in id_cols.items():
                if lab == "NAME" and not _is_doctor_header(c) and isinstance(df.at[i, c], str):
                    toks |= _name_tokens(df.at[i, c])
            patient_tokens.append(frozenset(toks))
        people = [_People(k, pt) for k, pt in zip(keys, patient_tokens)]

        def record(i, col, text, f):
            ctx = text[max(0, f.start - 40):f.start] + "⟦" + f.text + "⟧" + text[f.end:f.end + 40]
            rows.append({"record": i, "column": col, "label": f.label, "text": f.text, "context": ctx,
                         "found_by": "+".join(sorted(f.sources)), "confidence": round(f.confidence, 3),
                         "decision": f.decision, "reason": f.reason, "placeholder": f.placeholder,
                         "reviewer_decision": "",
                         "_features": f.features})

        # columns whose header says the whole cell is an identifier
        for c, lab in id_cols.items():
            for n, i in enumerate(df.index):
                v = df.at[i, c]
                if not isinstance(v, str) or not v.strip():
                    if pd.notna(v) and str(v).strip():
                        v = str(v)
                    else:
                        continue
                f = Finding(0, len(v), v, lab, {"column"}, 1.0, confidence=1.0, decision="mask",
                            role=("doctor" if _is_doctor_header(c) else "patient") if lab == "NAME" else None)
                out.at[i, c] = apply_placeholders(v, [f], vault, people[n])
                record(i, c, v, f)

        # free text: all three steps, with the row's known patient name as extra evidence
        for c in text_cols:
            vals = [v if isinstance(v, str) else "" for v in df[c]]
            for n, (i, v, fs) in enumerate(zip(df.index, vals, self.analyze(vals, patient_tokens))):
                if not v:
                    continue
                v = nfc(v)
                out.at[i, c] = apply_placeholders(v, fs, vault, people[n])
                for f in fs:
                    record(i, c, v, f)

        # every other text column: structured identifiers only
        for c in other_cols:
            for n, i in enumerate(df.index):
                v = df.at[i, c]
                if not isinstance(v, str):
                    continue
                fs = [f for f in find_structured(v) if f.prior >= 0.5]
                if not fs:
                    continue
                for f in fs:
                    f.confidence, f.decision = f.prior, ("mask" if f.prior >= 0.9 else "review")
                fs = _resolve_overlaps(fs)
                out.at[i, c] = apply_placeholders(v, fs, vault, people[n])
                for f in fs:
                    record(i, c, v, f)

        findings = pd.DataFrame(rows)
        if len(findings):
            findings.insert(0, "finding_id", range(1, len(findings) + 1))
        return out, findings, vault


# ---------------------------------------------------------------- column guessing

_ID_HEADERS = {
    "CURP": {"curp"}, "NSS": {"nss", "imss", "afiliacion"}, "RFC": {"rfc"},
    "PHONE": {"telefono", "tel", "celular", "cel", "movil", "whatsapp"},
    "EMAIL": {"correo", "email", "mail"},
    "ADDRESS": {"direccion", "domicilio", "calle", "colonia"},
    "BIRTHDATE": {"nacimiento", "nac", "dob"},
    "NAME": {"nombre", "nombres", "apellido", "apellidos", "paterno", "materno", "medico", "doctor"},
}
_TEXT_HEADERS = {"nota", "notas", "observaciones", "comentarios", "comentario", "motivo", "resumen", "evolucion",
                 "descripcion", "historia", "padecimiento", "texto"}


def _header_words(col: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", norm(str(col))))


def _is_doctor_header(col: str) -> bool:
    return bool(_header_words(col) & {"medico", "doctor"})


def guess_id_columns(df: pd.DataFrame) -> dict[str, str]:
    out = {}
    for c in df.columns:
        words = _header_words(c)
        if words & _TEXT_HEADERS:
            continue
        for lab, keys in _ID_HEADERS.items():
            if words & keys:
                out[c] = lab
                break
    return out


def guess_text_columns(df: pd.DataFrame, id_cols: dict[str, str]) -> list[str]:
    out = []
    for c in df.columns:
        if c in id_cols or not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue
        vals = [v for v in df[c] if isinstance(v, str)]
        if _header_words(c) & _TEXT_HEADERS or (vals and sum(map(len, vals)) / len(vals) > 40):
            out.append(c)
    return out
