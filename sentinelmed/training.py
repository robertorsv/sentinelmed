"""Training the confirmation classifier, measuring the pipeline, and learning from reviewers."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .classifier import DEFAULT_MODEL, Confirmer, backoff_variant, train_model
from .language import load_ner
from .pipeline import SentinelMed
from .synth import make_records
from .textutil import nfc
from .types import Finding

CORRECTIONS = DEFAULT_MODEL.parent / "corrections.jsonl"
SEEDS = {"train": 11, "val": 12, "test": 13, "hard": 14, "blind": 15, "final": 16}


def _overlap(f: Finding, spans, same_label: bool = True) -> float:
    """Share of the finding's characters that fall inside gold spans."""
    covered = sum(max(0, min(f.end, e) - max(f.start, s)) for s, e, lab in spans if not same_label or lab == f.label)
    return covered / max(1, f.end - f.start)


def build_training_set(records: list[dict], ner, augment: bool = False, seed: int = 0) -> tuple[list[dict], list[int]]:
    """Every candidate the detectors raise, labelled by the gold spans."""
    rng = random.Random(seed)
    sm = SentinelMed(use_classifier=False, ner=ner, use_ner=ner is not None)
    texts = [nfc(r["nota"]) for r in records]
    spans = ner.person_spans(texts) if ner else [()] * len(texts)
    X, y = [], []
    for r, text, per in zip(records, texts, spans):
        # with and without the name from the table, so the model works for both tables and loose notes
        for known in (frozenset(), frozenset(r["_patient_tokens"])):
            for f in sm.candidates(text, per, known):
                label = int(_overlap(f, r["_spans"]) >= 0.5)
                X.append(f.features)
                y.append(label)
                if augment and rng.random() < 0.6:
                    # the same candidate as it would look in an unfamiliar sentence, with an unfamiliar name
                    real_name = label == 1 and f.label == "NAME"
                    X.append(backoff_variant(f.features, unseen_name=real_name and rng.random() < 0.7,
                                             no_trigger=real_name and rng.random() < 0.7))
                    y.append(label)
    return X, y


def load_corrections(path: Path = CORRECTIONS) -> tuple[list[dict], list[int]]:
    X, y = [], []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            X.append(d["features"])
            y.append(d["label"])
    return X, y


def train(n_train: int = 4000, n_val: int = 800, out: Path = DEFAULT_MODEL, use_ner: bool = True,
          corrections: Path = CORRECTIONS, correction_weight: float = 10.0, max_missed: float = 0.003,
          verbose: bool = True) -> Confirmer:
    ner = load_ner() if use_ner else None
    X, y = build_training_set(make_records(n_train, SEEDS["train"]), ner, augment=True)
    Xc, yc = load_corrections(corrections)
    weights = [1.0] * len(X) + [correction_weight] * len(Xc)      # a reviewer's word counts for more
    model = train_model(X + Xc, y + yc, weights)

    # Choose the "leave it unmasked" threshold on fresh data: the highest value that
    # still lets through at most `max_missed` of the real identifiers.
    Xv, yv = build_training_set(make_records(n_val, SEEDS["val"]), ner)
    pv = model.predict_proba(Xv)[:, 1]
    pos = np.sort(pv[np.array(yv) == 1])
    # ... and never above 0.2: validation data resembles the training data more than real notes will.
    t_skip = float(min(0.2, pos[int(max_missed * len(pos))] - 1e-6))
    conf = Confirmer(model, t_skip=round(t_skip, 4), t_auto=0.9, meta={
        "trained_on": f"{n_train} synthetic records", "candidates": len(X), "positives": int(sum(y)),
        "reviewer_corrections": len(Xc), "ner": bool(ner), "validation_candidates": len(Xv),
    })
    conf.save(out)
    if verbose:
        neg = pv[np.array(yv) == 0]
        print(f"trained on {len(X)} candidates ({sum(y)} real identifiers) + {len(Xc)} reviewer corrections")
        print(f"skip threshold {conf.t_skip}: rejects {np.mean(neg < conf.t_skip):.1%} of false candidates "
              f"and {np.mean(pos < conf.t_skip):.2%} of real ones on validation data")
        print(f"saved {out}")
    return conf


def add_corrections(review_csv: Path, findings_jsonl: Path, store: Path = CORRECTIONS) -> int:
    """Read the reviewer's decisions (mask / keep) from a review CSV and store them for retraining.

    Only the feature dictionaries are stored: surrounding words and word endings, never the identifier.
    """
    review = pd.read_csv(review_csv, dtype=str).fillna("")
    feats = {}
    for line in Path(findings_jsonl).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        feats[str(d["finding_id"])] = d["features"]
    n = 0
    with open(store, "a", encoding="utf-8") as fh:
        for _, row in review.iterrows():
            verdict = row.get("reviewer_decision", "").strip().lower()
            if verdict not in {"mask", "keep"} or row["finding_id"] not in feats:
                continue
            fh.write(json.dumps({"features": feats[row["finding_id"]], "label": int(verdict == "mask")},
                                ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------- evaluation


def evaluate(records: list[dict], sm: SentinelMed, use_known: bool = False) -> dict:
    """Precision and recall on notes with gold labels.

    * recall: a gold identifier counts as protected only if every letter and digit of it is masked
    * precision: a masked span counts as correct if at least half of it is a real identifier
    """
    texts = [nfc(r["nota"]) for r in records]
    known = [frozenset(r["_patient_tokens"]) if use_known else frozenset() for r in records]
    results = sm.analyze(texts, known)
    gold_by, caught_by, pred_by, right_by = Counter(), Counter(), Counter(), Counter()
    leaked_records = flagged = total_findings = gold_chars = caught_chars = 0
    misses, false_alarms = [], []
    for r, text, fs in zip(records, texts, results):
        mask = np.zeros(len(text) + 1, dtype=bool)
        for f in fs:
            if f.masked:
                mask[f.start:f.end] = True
        leak = False
        for s, e, lab in r["_spans"]:
            gold_by[lab] += 1
            gold_chars += e - s
            caught_chars += int(mask[s:e].sum())
            # punctuation inside an identifier ("PÉREZ LÓPEZ, JUAN") does not have to be masked
            if all(mask[k] for k in range(s, e) if text[k].isalnum()):
                caught_by[lab] += 1
            else:
                leak = True
                misses.append((lab, text[s:e], text[max(0, s - 30):e + 20]))
        leaked_records += leak
        for f in fs:
            total_findings += 1
            flagged += f.decision in ("review", "skip-review")
            if f.masked:
                pred_by[f.label] += 1
                if _overlap(f, r["_spans"], same_label=False) >= 0.5:
                    right_by[f.label] += 1
                else:
                    false_alarms.append((f.label, f.text, text[max(0, f.start - 30):f.end + 20]))
    g, c, p, rt = sum(gold_by.values()), sum(caught_by.values()), sum(pred_by.values()), sum(right_by.values())
    per_label = {lab: {"gold": gold_by[lab], "recall": caught_by[lab] / gold_by[lab] if gold_by[lab] else None,
                       "masked": pred_by[lab], "precision": right_by[lab] / pred_by[lab] if pred_by[lab] else None}
                 for lab in sorted(set(gold_by) | set(pred_by))}
    return {
        "records": len(records), "gold": g, "recall": c / g, "precision": rt / p if p else 0.0,
        "char_recall": caught_chars / gold_chars, "missed": g - c, "false_alarms": p - rt,
        "records_with_leak": leaked_records / len(records),
        "review_share": flagged / total_findings if total_findings else 0.0,
        "per_label": per_label, "_misses": misses, "_false_alarms": false_alarms,
    }


def run_evaluation(n: int = 1000, verbose: bool = True, include_final: bool = False) -> dict:
    """The ablation in the README: each layer added in turn.

    The "final" set was scored once, before the last fixes, and that score is the
    one reported. Scoring it again after changing the code turns it into one more
    development set, so it is left out unless asked for.
    """
    ner = load_ner()
    sets = {"standard": make_records(n, SEEDS["test"]), "hard": make_records(n, SEEDS["hard"], level="hard"),
            "blind": make_records(n, SEEDS["blind"], level="blind")}
    if include_final:
        sets["final"] = make_records(n, SEEDS["final"], level="final")
    conf = Confirmer.load()
    layers = [
        ("1 rules only", dict(use_context=False, use_ner=False, use_classifier=False), False),
        ("2 + context and name lists", dict(use_context=True, use_ner=False, use_classifier=False), False),
        ("3 + language model (spaCy)", dict(use_context=True, ner=ner, use_classifier=False), False),
        ("4 + confirmation classifier", dict(use_context=True, ner=ner, confirmer=conf), False),
        ("5 + patient name from table", dict(use_context=True, ner=ner, confirmer=conf), True),
    ]
    out = defaultdict(dict)
    for name, kw, known in layers:
        sm = SentinelMed(**kw)
        for set_name, recs in sets.items():
            out[set_name][name] = evaluate(recs, sm, known)
            if verbose:
                m = out[set_name][name]
                print(f"{set_name:9s} {name:30s} recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
                      f"missed {m['missed']:4d}  false alarms {m['false_alarms']:4d}  "
                      f"records with a leak {m['records_with_leak']:.1%}  to review {m['review_share']:.1%}")
    return out
