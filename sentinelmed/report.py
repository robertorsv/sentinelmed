"""The review report: every finding with its confidence, so a person only checks the uncertain ones.

Both files contain original identifiers (a reviewer has to see them). They are
written next to the masked output and are meant to stay on the same computer.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

FLAGGED = ("review", "skip-review")
DECISION_TEXT = {
    "mask": "Masked",
    "review": "Masked — please confirm",
    "skip-review": "Left in place — please check",
    "skip": "Left in place",
}
COLUMNS = ["finding_id", "record", "column", "label", "text", "context", "found_by", "confidence", "decision",
           "reason", "placeholder", "reviewer_decision"]


def write_reports(findings: pd.DataFrame, outdir: Path, source_name: str, n_records: int) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if findings.empty:
        findings = pd.DataFrame(columns=COLUMNS + ["_features"])
    order = findings.assign(_flag=~findings["decision"].isin(FLAGGED)).sort_values(["_flag", "confidence"])
    order[COLUMNS].to_csv(outdir / "review.csv", index=False, encoding="utf-8-sig")
    with open(outdir / "findings.jsonl", "w", encoding="utf-8") as fh:
        for _, r in findings.iterrows():
            fh.write(json.dumps({"finding_id": int(r["finding_id"]), "features": r["_features"]}, ensure_ascii=False) + "\n")
    stats = {
        "records": n_records,
        "masked": int(findings["decision"].isin(("mask", "review")).sum()),
        "masked_review": int((findings["decision"] == "review").sum()),
        "left_review": int((findings["decision"] == "skip-review").sum()),
        "left": int((findings["decision"] == "skip").sum()),
    }
    (outdir / "review.html").write_text(_html(order, stats, source_name), encoding="utf-8")
    return stats


def _context_html(ctx: str) -> str:
    before, _, rest = ctx.partition("⟦")
    hit, _, after = rest.partition("⟧")
    return f"…{html.escape(before)}<mark>{html.escape(hit)}</mark>{html.escape(after)}…"


def _html(f: pd.DataFrame, stats: dict, source_name: str) -> str:
    flagged = f[f["decision"].isin(FLAGGED)]
    by_label = (f[f["decision"].isin(("mask", "review"))].groupby("label").size().sort_values(ascending=False)
                if len(f) else pd.Series(dtype=int))
    review_by_label = flagged.groupby("label").size() if len(flagged) else pd.Series(dtype=int)

    rows = []
    for _, r in flagged.iterrows():
        kind = "kept" if r["decision"] == "skip-review" else "masked"
        outcome = html.escape(r["placeholder"]) if r["placeholder"] else "left in place"
        why = html.escape(r["reason"]) if r["reason"] else "found by " + html.escape(r["found_by"])
        rows.append(
            f"<tr class='{kind}'><td class='num'>{r['finding_id']}</td><td>{html.escape(str(r['record']))}<br>"
            f"<span class='dim'>{html.escape(str(r['column']))}</span></td><td><span class='tag'>{r['label']}</span></td>"
            f"<td class='ctx'>{_context_html(r['context'])}</td><td class='num'>{r['confidence']:.2f}</td>"
            f"<td>{DECISION_TEXT[r['decision']]}<br><span class='dim'>{outcome} · {why}</span></td></tr>")
    label_rows = "".join(
        f"<tr><td><span class='tag'>{lab}</span></td><td class='num'>{n}</td>"
        f"<td class='num'>{int(review_by_label.get(lab, 0))}</td></tr>" for lab, n in by_label.items())
    empty = "<p class='ok'>Nothing needs a second look in this file.</p>" if not rows else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SentinelMed review — {html.escape(source_name)}</title>
<style>
:root {{ --bg:#f6f7f5; --card:#fff; --ink:#1c2422; --dim:#5d6966; --line:#dfe4e1; --accent:#0d6b5e; --warn:#a15c00;
        --mark:#ffe08a; --markink:#3a2c00; --warnbg:#fff6e5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#121716; --card:#1a2120; --ink:#e7ecea; --dim:#9aa7a3; --line:#2b3533;
        --accent:#5fc9b6; --warn:#f0b35a; --mark:#6b5200; --markink:#fff3c9; --warnbg:#2a2213; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:28px 16px 60px; }}
h1 {{ font-size:22px; margin:0 0 2px; }} h2 {{ font-size:16px; margin:32px 0 10px; }}
.dim {{ color:var(--dim); font-size:13px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.tile b {{ display:block; font-size:26px; font-variant-numeric:tabular-nums; }}
.tile.attn b {{ color:var(--warn); }}
.note {{ background:var(--warnbg); border:1px solid var(--line); border-left:4px solid var(--warn); border-radius:8px;
         padding:10px 14px; margin:16px 0; }}
.scroll {{ overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; }}
th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--dim); }}
tr:last-child td {{ border-bottom:0; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.tag {{ font:600 12px ui-monospace,Menlo,Consolas,monospace; color:var(--accent); }}
.ctx {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; min-width:300px; }}
mark {{ background:var(--mark); color:var(--markink); padding:1px 3px; border-radius:3px; }}
tr.kept td:first-child {{ box-shadow:inset 3px 0 0 var(--warn); }}
.ok {{ color:var(--accent); font-weight:600; }}
table.small {{ width:auto; min-width:320px; }}
</style></head><body><main>
<h1>SentinelMed review report</h1>
<div class="dim">{html.escape(source_name)} · {stats['records']} records</div>
<div class="note"><b>This report shows original identifiers.</b> Keep it on this computer and delete it when the review is done.
Do not share it together with the masked file.</div>
<div class="tiles">
  <div class="tile"><b>{stats['masked']}</b>identifiers masked</div>
  <div class="tile attn"><b>{stats['masked_review']}</b>masked, please confirm</div>
  <div class="tile attn"><b>{stats['left_review']}</b>left in place, please check</div>
  <div class="tile"><b>{stats['left']}</b>candidates rejected with a stated reason</div>
</div>
<h2>Needs a person ({len(flagged)})</h2>
<p class="dim">Least certain first. A bar on the left marks text that was <b>not</b> masked. To correct the tool, open
<code>review.csv</code>, write <code>mask</code> or <code>keep</code> in the <code>reviewer_decision</code> column and run
<code>sentinelmed learn</code>.</p>
{empty}
{"<div class='scroll'><table><thead><tr><th>#</th><th>Record</th><th>Type</th><th>Text in context</th><th>Confidence</th><th>Decision</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" if rows else ""}
<h2>Masked by type</h2>
<div class="scroll" style="display:inline-block"><table class="small"><thead><tr><th>Type</th><th>Masked</th><th>Flagged</th></tr></thead>
<tbody>{label_rows or "<tr><td colspan=3 class='dim'>Nothing was masked.</td></tr>"}</tbody></table></div>
<p class="dim">The full list, including confident findings, is in <code>review.csv</code>.</p>
</main></body></html>
"""
