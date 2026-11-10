"""Command line: sentinelmed demo | mask | train | evaluate | learn"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=object)
    return pd.read_csv(path, dtype=object, keep_default_na=False, encoding="utf-8-sig")


def cmd_demo(a) -> None:
    from .synth import make_records

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = make_records(a.n, seed=a.seed, level="hard" if a.hard else "std")
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    df.to_csv(out / "pacientes_sinteticos.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} synthetic records (all invented) to {out / 'pacientes_sinteticos.csv'}")


def cmd_mask(a) -> None:
    from .pipeline import SentinelMed, Vault, _People, apply_placeholders
    from .report import write_reports
    from .textutil import nfc

    src, out = Path(a.input), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    deny = Path(a.denylist).read_text(encoding="utf-8").splitlines() if a.denylist else None
    sm = SentinelMed(use_ner=not a.no_ner, denylist=deny, model_path=a.model)
    if sm.confirmer is None:
        print("! No trained classifier found; every candidate will be masked. Run `sentinelmed train` first.")
    if not a.no_ner and sm.ner is None:
        print("! spaCy model not available; running without the language-model layer "
              "(pip install spacy && python -m spacy download es_core_news_sm).")

    if src.suffix.lower() == ".txt":
        text = nfc(src.read_text(encoding="utf-8"))
        vault = Vault()
        findings = sm.analyze([text])[0]
        masked = apply_placeholders(text, findings, vault, _People("0"))
        (out / f"{src.stem}.masked.txt").write_text(masked, encoding="utf-8")
        rows = [{"finding_id": i + 1, "record": 0, "column": "text", "label": f.label, "text": f.text,
                 "context": text[max(0, f.start - 40):f.start] + "⟦" + f.text + "⟧" + text[f.end:f.end + 40],
                 "found_by": "+".join(sorted(f.sources)), "confidence": round(f.confidence, 3), "decision": f.decision,
                 "reason": f.reason, "placeholder": f.placeholder, "reviewer_decision": "", "_features": f.features}
                for i, f in enumerate(findings)]
        fdf, n = pd.DataFrame(rows), 1
    else:
        df = _read_table(src)
        id_cols = dict(kv.split("=") for kv in a.id_cols.split(",")) if a.id_cols else None
        text_cols = a.text_cols.split(",") if a.text_cols else None
        masked_df, fdf, vault = sm.mask_table(df, text_cols=text_cols, id_cols=id_cols)
        if src.suffix.lower() in (".xlsx", ".xls"):
            masked_df.to_excel(out / f"{src.stem}.masked.xlsx", index=False)
        else:
            masked_df.to_csv(out / f"{src.stem}.masked.csv", index=False, encoding="utf-8-sig")
        n = len(df)

    stats = write_reports(fdf, out, src.name, n)
    if a.save_key:
        vault.key_table().to_csv(out / "key.csv", index=False, encoding="utf-8-sig")
        print("! key.csv links every placeholder to the original value. Store it apart from the masked file, or delete it.")
    print(f"{stats['masked']} identifiers masked in {n} record(s); "
          f"{stats['masked_review'] + stats['left_review']} need a person "
          f"({stats['masked_review']} masked, {stats['left_review']} left in place).")
    print(f"Output in {out}/ — open review.html first.")


def cmd_train(a) -> None:
    from .training import train

    train(n_train=a.n, use_ner=not a.no_ner)


def cmd_evaluate(a) -> None:
    from .training import run_evaluation

    run_evaluation(a.n, include_final=a.final)


def cmd_learn(a) -> None:
    from .training import add_corrections, train

    n = add_corrections(Path(a.review), Path(a.findings) if a.findings else Path(a.review).with_name("findings.jsonl"))
    print(f"{n} reviewer decision(s) added.")
    if n and not a.no_retrain:
        train(use_ner=not a.no_ner)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sentinelmed", description="Local de-identification of Mexican clinical data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="write a synthetic patient table to try the tool on")
    s.add_argument("--out", default="demo")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--hard", action="store_true", help="unfamiliar names and sentence patterns")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("mask", help="mask a .csv, .xlsx or .txt file")
    s.add_argument("input")
    s.add_argument("--out", default="sentinelmed_out")
    s.add_argument("--text-cols", help="free-text columns, comma separated (default: guessed)")
    s.add_argument("--id-cols", help="identifier columns as name=LABEL, comma separated (default: guessed from headers)")
    s.add_argument("--denylist", help="text file with one extra term per line that must always be masked")
    s.add_argument("--no-ner", action="store_true", help="skip the spaCy language-model layer")
    s.add_argument("--model", help="path to a trained classifier (default: models/confirm.joblib)")
    s.add_argument("--save-key", action="store_true", help="also write key.csv (placeholder -> original value)")
    s.set_defaults(fn=cmd_mask)

    s = sub.add_parser("train", help="train the confirmation classifier on synthetic records")
    s.add_argument("--n", type=int, default=4000)
    s.add_argument("--no-ner", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="precision and recall, layer by layer, on synthetic test sets")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--final", action="store_true", help="also score the final held-out set (see README before using)")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("learn", help="add a reviewer's decisions from review.csv and retrain")
    s.add_argument("review")
    s.add_argument("--findings", help="findings.jsonl written next to review.csv (default: same folder)")
    s.add_argument("--no-retrain", action="store_true")
    s.add_argument("--no-ner", action="store_true")
    s.set_defaults(fn=cmd_learn)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
