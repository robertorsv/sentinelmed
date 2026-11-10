"""The confirmation classifier: a second reviewer for every candidate.

It never sees the raw identifier. It sees how the candidate was found, what it
looks like (shape, list membership, word endings) and the words around it, and
returns the probability that the candidate really is an identifier.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import lexicons as L
from .textutil import Tok
from .types import Finding

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "confirm.joblib"


# Words that only ever name a place ("Mexicali", "Sonora"), not ones shared with people ("León", "Juárez").
_PLACE_WORDS = frozenset(w for place in (L.CITIES | L.STATES) for w in place.replace(".", " ").split()
                         if len(w) > 3 and w not in L.FIRST_NAMES and w not in L.SURNAMES
                         and w not in {"ciudad", "estado", "nuevo", "baja"})


def _shape(w: str) -> str:
    return re.sub(r"\d+", "0", w)


_PREPOSITIONS = {"a", "de", "del", "con", "por", "en", "al", "para"}


def _word_class(t: Tok | None, edge: str) -> str:
    """A coarse description of a neighbouring word that still means something for an unseen word."""
    if t is None:
        return edge
    w = t.norm
    if not t.is_alpha:
        return "0" if w[0].isdigit() else w
    for name, lex in (("prep", _PREPOSITIONS), ("kin", L.KIN_TERMS), ("doctor", L.DOCTOR_TITLES),
                      ("title", L.PERSON_TITLES), ("patient", L.PATIENT_TRIGGERS), ("verb", L.PERSON_VERBS),
                      ("objverb", L.OBJECT_VERBS), ("label", L.PERSON_LABELS), ("unit", L.UNITS),
                      ("eponym", L.EPONYM_HEADS), ("inst", L.INSTITUTION_WORDS), ("paper", L.NUMBER_PAPERWORK),
                      ("phone", L.PHONE_TRIGGERS), ("nss", L.NSS_TRIGGERS), ("stop", L.STOPWORDS)):
        if w in lex:
            return name
    return "Other" if t.is_cap else "other"


_NAME_STRONG = ("src_context", "src_gazetteer", "src_propagated")


def _name_interactions(x: dict) -> None:
    """Features a linear model cannot build by itself. Recomputed after any change to the sources."""
    weak = not any(k in x for k in _NAME_STRONG)          # found only by shape, NER or extension
    x["weak_only"] = int(weak)
    x["weak_single"] = int(weak and x.get("one_tok", 0) == 1)
    x["ner_guess_single"] = int(x.get("one_tok", 0) == 1 and "src_ner" in x and "src_shape" not in x and weak)
    x["amb_no_cue"] = int(x.get("all_amb_one", 0) == 1 and "src_context" not in x and "src_propagated" not in x)
    x["multi_cap"] = int(x.get("n_tok", 1) >= 2 and x.get("frac_cap", 0) == 1.0)
    x["unknown_name"] = int(x.get("frac_gaz", 0) == 0)


def backoff_variant(x: dict, unseen_name: bool = False, no_trigger: bool = False) -> dict:
    """A copy of a feature dict as it would look in a sentence pattern never seen in training.

    Synthetic training text is far more regular than real notes: nearly every
    name follows a familiar word and is in the name list. Left alone, the model
    learns "no trigger word, so not a name" and rejects real names in new text.
    These variants remove the exact neighbouring words, and optionally the
    name-list hit and the trigger, so the model must also learn from what carries
    over. Used only to augment training data.
    """
    y = {k: v for k, v in x.items() if not (k in ("prev1", "next1") or k.endswith(("|prev1", "|next1", "|prev2", "|next2")))}
    if y.get("label") != "NAME":
        return y

    def drop(source: str) -> None:
        for k in [k for k in y if k.endswith(source)]:
            del y[k]
        y["n_src"] = max(1, y.get("n_src", 1) - 1)

    others = lambda keep: any(k in y for k in ("src_context", "src_gazetteer", "src_ner", "src_shape", "src_propagated")
                              if k != keep)
    if unseen_name and "src_gazetteer" in y and others("src_gazetteer"):
        drop("src_gazetteer")
        y.update({"frac_first": 0.0, "frac_sur": 0.0, "frac_gaz": 0.0, "frac_amb": 0.0, "all_amb_one": 0})
    if no_trigger and "src_context" in y and others("src_context"):
        drop("src_context")
        y.update({"trig_kin": 0, "trig_doctor": 0, "trig_patient": 0, "trig_title": 0})
        y.pop("role", None)
        for k in [k for k in y if k.startswith("NAME|") and k.endswith("_class")]:
            del y[k]
    _name_interactions(y)
    return y


def rejection_reason(f: Finding) -> str | None:
    """Why a candidate may be left unmasked. No reason, no rejection.

    The classifier alone is not allowed to wave a candidate through: absence of
    evidence that something is an identifier is not evidence that it is safe.
    """
    x, lab = f.features, f.label
    if lab == "NAME":
        if x.get("frac_drug", 0) > 0:
            return "medicine name"
        if x.get("frac_med", 0) > 0 or x.get("prev_eponym_head"):
            return "disease, sign or test named after a person"
        if x.get("next_unit") or x.get("next_dose"):
            return "followed by a dose"
        if x.get("frac_inst", 0) > 0 or x.get("prev_inst"):
            return "part of an institution name"
        if x.get("amb_no_cue"):
            return "ordinary word that is also a surname, with nothing announcing a name"
        if x.get("frac_place", 0) == 1.0:
            return "place name"
        if x.get("next_colon"):
            return "field label"
        if x.get("ner_guess_single"):
            return "single word at the start of a sentence, suggested only by the language model"
        return None
    if lab == "PHONE":
        if x.get("PHONE|paperwork"):
            return "number follows a paperwork word (folio, orden, muestra)"
        if not x.get("PHONE|has_sep") and not x.get("PHONE|trig_phone"):
            return "bare 10-digit number with no phone word nearby"
        return None
    if lab == "NSS":
        if x.get("NSS|paperwork"):
            return "number follows a paperwork word (folio, orden, muestra)"
        if x.get("NSS|valid") == "n" and not x.get("NSS|trig_nss"):
            return "fails the NSS check digit and no IMSS word nearby"
        if not x.get("NSS|trig_nss") and not x.get("NSS|has_sep"):
            return "bare 11-digit number with no IMSS word nearby"
        return None
    if lab == "ADDRESS":
        if x.get("addr_ambiguous_type") and not x.get("addr_number"):
            return "street word used as an ordinary word"
        return None
    return None


def featurize(text: str, toks: list[Tok], starts: list[int], f: Finding, mode: str) -> dict:
    i0 = bisect_left(starts, f.start)
    i1 = max(i0, bisect_left(starts, f.end) - 1)
    inner = toks[i0:i1 + 1]
    words = [t for t in inner if t.is_alpha and t.norm not in L.NAME_PARTICLES]
    prev = [_shape(toks[k].norm) for k in range(i0 - 1, max(-1, i0 - 4), -1)]
    nxt = [_shape(toks[k].norm) for k in range(i1 + 1, min(len(toks), i1 + 3))]
    # the words before the candidate, within the same sentence
    prev_words, k = set(), i0 - 1
    while k >= 0 and len(prev_words) < 5 and i0 - k <= 8:
        if toks[k].text in {".", ";"} and not (k > 0 and toks[k - 1].norm in L.ABBREVIATIONS | {"no", "c", "p"}):
            break
        if toks[k].is_alpha:
            prev_words.add(toks[k].norm)
        k -= 1

    lab = f.label
    x: dict = {"label": lab, "mode": mode, "n_src": len(f.sources)}
    for s in f.sources:
        x[f"src_{s}"] = 1
        x[f"{lab}|src_{s}"] = 1
        x[f"{mode}|src_{s}"] = 1
    if f.valid is not None:
        x[f"{lab}|valid"] = "y" if f.valid else "n"
    p1 = prev[0] if prev else "<s>"
    n1 = nxt[0] if nxt else "</s>"
    x["prev1"], x["next1"] = p1, n1
    x[f"{lab}|prev1"], x[f"{lab}|next1"] = p1, n1
    pc = _word_class(toks[i0 - 1] if i0 > 0 else None, "<s>")
    nc = _word_class(toks[i1 + 1] if i1 + 1 < len(toks) else None, "</s>")
    x[f"{lab}|prev_class"], x[f"{lab}|next_class"] = pc, nc
    if i0 > 1:
        x[f"{lab}|prev2_class"] = _word_class(toks[i0 - 2], "<s>")
    if len(prev) > 1:
        x[f"{lab}|prev2"] = prev[1]
    if len(nxt) > 1:
        x[f"{lab}|next2"] = nxt[1]

    if lab == "NAME" and words:
        n = len(words)

        def frac(lex):
            return sum(t.norm in lex for t in words) / n

        gaz = sum((t.norm in L.FIRST_NAMES or t.norm in L.SURNAMES) for t in words) / n
        amb = frac(L.AMBIGUOUS)
        x.update({
            "n_tok": min(n, 5), "one_tok": int(n == 1),
            "frac_first": frac(L.FIRST_NAMES), "frac_sur": frac(L.SURNAMES), "frac_gaz": gaz,
            "frac_drug": frac(L.DRUGS), "frac_med": frac(L.MEDICAL_TERMS), "frac_inst": frac(L.INSTITUTION_WORDS),
            "frac_amb": amb, "frac_cap": sum(t.is_cap for t in words) / n,
            "all_amb_one": int(n == 1 and amb == 1.0),
            "sent_start": int(p1 in {"<s>", "."} and not (len(prev) > 1 and prev[1] in L.DOCTOR_TITLES | L.PERSON_TITLES)),
            "next_unit": int(n1 in L.UNITS), "next_digit": int(n1[:1].isdigit()),
            "next_dose": int(n1[:1].isdigit() and len(nxt) > 1 and nxt[1] in L.UNITS),      # "Bisoprolol 500 mg"
            "next_lower": int(i1 + 1 < len(toks) and toks[i1 + 1].is_alpha and not toks[i1 + 1].is_cap),
            "trig_kin": int(bool(prev_words & L.KIN_TERMS)), "trig_doctor": int(bool(prev_words & L.DOCTOR_TITLES)),
            "trig_patient": int(bool(prev_words & L.PATIENT_TRIGGERS)), "trig_title": int(bool(prev_words & L.PERSON_TITLES)),
            "prev_inst": int(bool(prev_words & L.INSTITUTION_WORDS)),
            "prev_eponym_head": int(bool(set(prev[:2]) & L.EPONYM_HEADS)),
            "frac_place": frac(_PLACE_WORDS), "frac_upper": sum(t.text.isupper() for t in words) / n,
            "next_verb": int(n1 in L.PERSON_VERBS), "next_colon": int(n1 == ":"),
            "suffix3": words[0].norm[-3:], "suffix4": words[0].norm[-4:], "last_suffix3": words[-1].norm[-3:],
        })
        if f.role:
            x["role"] = f.role
        _name_interactions(x)
    elif lab in ("PHONE", "NSS"):
        x.update({
            f"{lab}|has_sep": int(bool(re.search(r"[ .()-]", f.text))),
            f"{lab}|trig_phone": int(bool(prev_words & L.PHONE_TRIGGERS)),
            f"{lab}|trig_nss": int(bool(prev_words & L.NSS_TRIGGERS)),
            f"{lab}|paperwork": int(bool(prev_words & L.NUMBER_PAPERWORK)),
            f"{lab}|plus": int(f.text.startswith("+")),
        })
    elif lab == "ADDRESS":
        x.update({
            "addr_number": int(bool(re.search(r"\d", f.text))),
            "addr_colonia": int(any(t.norm in L.COLONIA_WORDS for t in inner)),
            "addr_ambiguous_type": int(bool(inner) and inner[0].norm in L.AMBIGUOUS_STREET_TYPES),
            "addr_n_tok": min(len(inner), 12),
        })
    return x


def train_model(X: list[dict], y: list[int], sample_weight=None) -> Pipeline:
    model = Pipeline([
        ("vec", DictVectorizer()),
        # Positives weigh more: a missed identifier is the worse error.
        ("clf", LogisticRegression(C=1.0, class_weight={0: 1.0, 1: 3.0}, max_iter=3000)),
    ])
    model.fit(X, y, clf__sample_weight=sample_weight)
    return model


class Confirmer:
    """Scores candidates and turns the score into one of four decisions.

    ==============  ========================  ================================
    probability     decision                  what happens
    ==============  ========================  ================================
    >= t_auto       mask                      masked
    >= t_skip       review                    masked, and flagged for a person
    >= t_ignore     skip-review               left in place, flagged for a person
    <  t_ignore     skip                      left in place
    ==============  ========================  ================================
    """

    def __init__(self, model: Pipeline, t_skip: float = 0.2, t_auto: float = 0.9, t_ignore: float = 0.02,
                 meta: dict | None = None):
        self.model, self.t_skip, self.t_auto, self.t_ignore, self.meta = model, t_skip, t_auto, t_ignore, meta or {}

    def score(self, feats: list[dict]) -> list[float]:
        if not feats:
            return []
        return [float(p) for p in self.model.predict_proba(feats)[:, 1]]

    def decide(self, p: float) -> str:
        if p >= self.t_auto:
            return "mask"
        if p >= self.t_skip:
            return "review"
        return "skip-review" if p >= self.t_ignore else "skip"

    def save(self, path: Path = DEFAULT_MODEL) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "t_skip": self.t_skip, "t_auto": self.t_auto, "t_ignore": self.t_ignore,
                     "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL) -> "Confirmer":
        d = joblib.load(path)
        return cls(d["model"], d["t_skip"], d["t_auto"], d.get("t_ignore", 0.02), d.get("meta"))
