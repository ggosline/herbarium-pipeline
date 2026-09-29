"""
Second opinion: check a sheet against the flora's descriptions of each candidate.

The classifier says which species a sheet *looks like*; it cannot say why, and
it cannot name a species it never trained on. This script shows Claude the
sheet, the candidates (the re-ranked top-5 from `flora_link.py rerank` plus any
blind-spot species it suggested), and for each candidate the published
description and plate from the flora wiki. Claude records what is measurable on
the sheet, then says for each candidate whether the description is consistent
with it, citing the characters that agree or conflict.

What comes out, per specimen:
  observed            what is visible and measurable (blade size if a scale is
                      on the sheet, vein pairs, glands, indumentum, fertile parts)
  per-candidate       consistent / inconsistent / cannot_tell, with the
                      supporting and conflicting characters
  best_match          a candidate, or "none"
  outside_candidates  no candidate fits: a species outside the list, or an
                      undescribed one. These are the sheets worth a botanist's
                      time first.
  label_name_seen     any determination legible on the sheet. Claude is told
                      not to use it; recording it lets you audit whether it did.

Candidates are shown alphabetically and without the classifier's scores, so the
comparison is made from the descriptions, not anchored on the model.

Selection (default): indets whose re-ranked top-1 is below --max-prob, or that
the re-rank changed, or that carry a blind-spot candidate; lowest top-1 first.
--select verified samples flora-verified sheets instead, to measure agreement
with a known answer before trusting it on indets.

Costs real money: each sheet is one Claude Opus 5.5 request with the specimen
image, up to ~8 candidate descriptions and plates. Run --dry-run first; it
builds every request, writes an example prompt, and estimates the cost without
calling the API. Results append to second_opinion.jsonl, so an interrupted run
resumes where it stopped.

Needs `pip install anthropic` (not a pipeline dependency) and an API key in
ANTHROPIC_API_KEY or an `ant auth login` profile.

Usage:
  python second_opinion.py \\
      --wiki      E:/flora-ocr/wiki \\
      --reranked  E:/Diospyros/review/flora_link/reranked.csv \\
      --images    E:/Diospyros/images_cropped \\
      --out       E:/Diospyros/review/flora_link \\
      --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

from flora_link import NameIndex, Page, clean_md, load_pages

Image.MAX_IMAGE_PIXELS = None

MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.00, 20.00        # $ per million tokens, claude-opus-5-5
SHEET_PIXELS = 1_150_000                 # the API downsamples beyond ~1.15 MP anyway
PLATE_PIXELS = 600_000
MAX_DESC_CHARS = 8000
DESC_SECTIONS = re.compile(r"^(description|diagnosis|taxonomic affinities)$", re.I)

SYSTEM = """\
You are a taxonomist of tropical African plants examining one digitised \
herbarium sheet. A machine-vision classifier has proposed candidate species. \
For each candidate you are given its description from a published flora or \
protologue and, where one exists, the flora's botanical plate. Your task is to \
check each description against what is actually visible on this sheet.

Work in this order:
1. Record what the sheet shows. Measure leaf blades in cm only if a ruler or \
scale bar is visible; otherwise say there is no scale and give proportions \
(length:width, petiole relative to blade) instead. Count lateral vein pairs on \
the clearest leaf. Note glands, indumentum, and which fertile parts are present \
(male or female flowers, fruits, accrescent calyx).
2. Compare each candidate's description with those observations. A measurable \
character outside the described range, including parenthetical extremes, is \
strong evidence against. A character you cannot see is not evidence either \
way: use cannot_tell rather than guessing.
3. Choose the best match, or "none" if no candidate is consistent. Set \
outside_candidates when the sheet conflicts with every candidate on at least \
one clear character: that is how a species missing from the list, or an \
undescribed one, shows up, and it is a useful result, not a failure.

Judge from morphology only. Do not use names written on labels or \
determination slips; if one is legible, report it in label_name_seen and \
otherwise ignore it. Candidates are listed alphabetically; the order carries no \
information. Name characters specifically ("blade 18 cm, description gives \
10-15 (-26) cm"), not in general terms."""

SCHEMA = {
    "type": "object",
    "properties": {
        "observed": {
            "type": "object",
            "properties": {
                "organs_present": {"type": "array", "items": {"type": "string"}},
                "scale_available": {"type": "boolean"},
                "leaf_blade_size": {"type": "string"},
                "leaf_shape_apex_base": {"type": "string"},
                "lateral_vein_pairs": {"type": "string"},
                "glands": {"type": "string"},
                "indumentum": {"type": "string"},
                "reproductive": {"type": "string"},
                "other": {"type": "string"},
            },
            "required": ["organs_present", "scale_available", "leaf_blade_size",
                         "leaf_shape_apex_base", "lateral_vein_pairs", "glands",
                         "indumentum", "reproductive", "other"],
            "additionalProperties": False,
        },
        "label_name_seen": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "verdict": {"type": "string",
                                "enum": ["consistent", "inconsistent", "cannot_tell"]},
                    "supporting": {"type": "array", "items": {"type": "string"}},
                    "conflicting": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "verdict", "supporting", "conflicting"],
                "additionalProperties": False,
            },
        },
        "best_match": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "outside_candidates": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["observed", "label_name_seen", "candidates", "best_match",
                 "confidence", "outside_candidates", "summary"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def encode_image(path: Path, max_pixels: int) -> tuple[str, str, int]:
    """JPEG, downscaled to max_pixels. Returns (media_type, b64, est. tokens)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, (max_pixels / (im.width * im.height)) ** 0.5)
        if scale < 1:
            im = im.resize((int(im.width * scale), int(im.height * scale)),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        tokens = im.width * im.height // 750
    return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode(), tokens


def description_text(pg: Page | None, name: str) -> tuple[str, str]:
    """(text, provenance) for a candidate. Never silently truncated."""
    if pg is None:
        return ("No description available: this species is not treated in the "
                "regional flora (it may occur outside Lower Guinea).", "none")
    if pg.kind == "placeholder":
        return ("No description available: the flora lists this species as "
                "accepted for the region but has not described it.", "placeholder")
    keep, take = [], False
    for line in pg.body.splitlines():
        h = re.match(r"^#{2,4}\s+(.*)$", line)
        if h:
            head = clean_md(h.group(1))
            take = bool(DESC_SECTIONS.match(head))
            if take:
                keep.append(f"[{head}]")
            continue
        # "— Maba Mualala Welw. ex Hiern, ..." synonymy lines: no morphology
        if take and line.strip() and not re.match(r"^\s*[—–-]\s", line):
            keep.append(clean_md(line))
    text = "\n".join(keep).strip() or "The wiki page has no description section."
    if len(text) > MAX_DESC_CHARS:
        text = text[:MAX_DESC_CHARS] + " [...description truncated]"
    src = pg.fm.get("source") or pg.fm.get("vol") or pg.kind
    return text, str(src)


def first_plate(pg: Page | None, wiki: Path) -> tuple[Path, str] | None:
    if pg is None:
        return None
    m = re.search(r"!\[[^\]]*\]\(\.\./sources/([^)]+)\)\s*\n+\s*\*([^\n]*)\*", pg.body)
    if not m:
        m = re.search(r"!\[[^\]]*\]\(\.\./sources/([^)]+)\)", pg.body)
    if not m:
        return None
    # wiki/sources is a symlink to ../ocr_output, which Windows may not follow
    path = wiki.parent / "ocr_output" / m.group(1)
    if not path.exists():
        path = wiki / "sources" / m.group(1)
    caption = clean_md(m.group(2)) if m.lastindex and m.lastindex >= 2 else ""
    return (path, caption) if path.exists() else None


def candidates_for(row: pd.Series, k: int) -> list[str]:
    names = [row.get(f"top{i}_name", "") for i in range(1, k + 1)]
    for part in str(row.get("blind_spot_candidates", "") or "").split(" | "):
        if part.strip():
            names.append(part.split(" (")[0].strip())
    seen, out = set(), []
    for n in names:
        if isinstance(n, str) and n and n not in seen:
            seen.add(n)
            out.append(n)
    return sorted(out)


def build_request(row, cands, pages_by_name, wiki: Path, images: Path, plates: bool):
    sheet = images / row["fname"]
    media, b64, tok = encode_image(sheet, SHEET_PIXELS)
    genus = cands[0].split()[0] if cands else ""
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
        {"type": "text", "text":
            f"Herbarium sheet of {genus}. Country: {row.get('countryCode') or 'unknown'}.\n\n"
            f"Candidates ({len(cands)}), with their flora descriptions:"},
    ]
    est_tokens, text_chars = tok, 0
    sent: dict[Path, str] = {}      # one plate often figures two species (PL. 19)
    for name in cands:
        pg = pages_by_name.get(name)
        desc, src = description_text(pg, name)
        block = f"\n### {name}\nSource: {src}\n{desc}"
        content.append({"type": "text", "text": block})
        text_chars += len(block)
        plate = first_plate(pg, wiki) if plates else None
        if plate and plate[0] in sent:
            content.append({"type": "text", "text":
                            f"Plate for {name}: the same plate shown above for {sent[plate[0]]}."})
        elif plate:
            sent[plate[0]] = name
            pmedia, pb64, ptok = encode_image(plate[0], PLATE_PIXELS)
            content.append({"type": "text", "text": f"Plate for {name}: {plate[1] or 'no caption'}"})
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": pmedia, "data": pb64}})
            est_tokens += ptok
    content.append({"type": "text", "text":
        "Examine the sheet and assess every candidate, following your instructions."})
    est_tokens += (text_chars + len(SYSTEM)) // 4
    return content, est_tokens


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_rows(rr: pd.DataFrame, args) -> pd.DataFrame:
    if args.fnames:
        return rr[rr["fname"].isin(args.fnames)]
    if args.select == "verified":
        v = pd.read_csv(args.out / "flora_verified.csv", dtype=str).fillna("")
        v = v[v["model_class"].ne("")]
        # model_class: the flora's determination under the name the candidates use
        pick = rr[rr["fname"].isin(v["fname"])].merge(
            v[["fname", "model_class"]], on="fname")
        return pick.sample(frac=1, random_state=0)
    ind = rr[rr["indet"]]
    mask = (ind["top1_prob"] < args.max_prob) | ind["changed"] | \
        ind["blind_spot_candidates"].ne("")
    return ind[mask].sort_values("top1_prob")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def call_claude(client, content, effort: str):
    import anthropic
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"},
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            # On a safety decline, re-run on Anthropic's recommended fallback
            # model instead of returning the refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.BadRequestError as e:
        return {"error": f"bad request: {e.message}"}
    except anthropic.RateLimitError:
        return {"error": "rate limited after SDK retries"}
    except anthropic.APIStatusError as e:
        return {"error": f"API {e.status_code}: {e.message}"}
    except anthropic.APIConnectionError:
        return {"error": "connection error"}

    rec = {"model_served": resp.model, "stop_reason": resp.stop_reason,
           "request_id": resp._request_id,
           "input_tokens": resp.usage.input_tokens,
           "output_tokens": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        rec["error"] = "refused"
        return rec
    if resp.stop_reason == "max_tokens":
        rec["error"] = "hit max_tokens"
        return rec
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        rec["result"] = json.loads(text)
    except json.JSONDecodeError:
        rec["error"] = "unparseable output"
    return rec


def flatten(rec: dict) -> dict:
    r = rec.get("result") or {}
    by = {c["name"]: c for c in r.get("candidates", [])}
    return {
        "fname": rec["fname"],
        "countryCode": rec.get("countryCode", ""),
        "classifier_top1": rec.get("classifier_top1", ""),
        "classifier_prob": rec.get("classifier_prob", ""),
        "known_answer": rec.get("known_answer", ""),
        "best_match": r.get("best_match", ""),
        "confidence": r.get("confidence", ""),
        "outside_candidates": r.get("outside_candidates", ""),
        "consistent": "; ".join(n for n, c in by.items() if c["verdict"] == "consistent"),
        "inconsistent": "; ".join(n for n, c in by.items() if c["verdict"] == "inconsistent"),
        "cannot_tell": "; ".join(n for n, c in by.items() if c["verdict"] == "cannot_tell"),
        "label_name_seen": r.get("label_name_seen", ""),
        "summary": r.get("summary", ""),
        "error": rec.get("error", ""),
        "cost_usd": round(rec.get("input_tokens", 0) * PRICE_IN / 1e6
                          + rec.get("output_tokens", 0) * PRICE_OUT / 1e6, 4),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wiki", type=Path, required=True, help="flora-ocr wiki/ directory.")
    p.add_argument("--reranked", type=Path, required=True,
                   help="reranked.csv from `flora_link.py rerank`.")
    p.add_argument("--images", type=Path, required=True, help="Directory of sheet images.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--select", choices=["indets", "verified"], default="indets")
    p.add_argument("--fnames", nargs="*", help="Explicit sheets (overrides --select).")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-prob", type=float, default=0.5,
                   help="indets: include sheets whose re-ranked top-1 is below this.")
    p.add_argument("--top-k", type=int, default=5, help="Re-ranked candidates to include.")
    p.add_argument("--no-plates", action="store_true", help="Leave out flora plates.")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="Build requests and estimate cost; no API calls.")
    args = p.parse_args()

    rr = pd.read_csv(args.reranked, dtype=str, keep_default_na=False)
    rr["indet"] = rr["indet"].str.lower().eq("true")
    rr["changed"] = rr["changed"].str.lower().eq("true")
    rr["top1_prob"] = pd.to_numeric(rr["top1_prob"], errors="coerce").fillna(0)

    out_jsonl = args.out / "second_opinion.jsonl"
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if "result" in rec:
                done.add(rec["fname"])
    rows = select_rows(rr, args)
    rows = rows[~rows["fname"].isin(done)]
    rows = rows[rows["fname"].map(lambda f: (args.images / f).exists())].head(args.limit)
    print(f"{len(rows)} sheets selected ({len(done)} already done)")
    if rows.empty:
        return 0

    genera = {n.split()[0] for r in rows.itertuples() for n in
              candidates_for(pd.Series(r._asdict()), args.top_k)}
    pages = load_pages(args.wiki, genera)
    index = NameIndex(load_pages(args.wiki, None))
    pages_by_name: dict[str, Page] = {}

    def page_for(name: str) -> Page | None:
        if name not in pages_by_name:
            stem = index.resolve(name)[0]
            pages_by_name[name] = pages.get(stem) if stem else None
        return pages_by_name[name]

    jobs, est_total = [], 0
    for _, row in rows.iterrows():
        cands = candidates_for(row, args.top_k)
        for c in cands:
            page_for(c)
        content, est = build_request(row, cands, pages_by_name, args.wiki,
                                     args.images, not args.no_plates)
        jobs.append((row, cands, content))
        est_total += est

    args.out.mkdir(parents=True, exist_ok=True)
    example = args.out / "second_opinion_prompt_example.md"
    ex_row, ex_cands, ex_content = jobs[0]
    example.write_text(
        f"# Example request — {ex_row['fname']}\n\n## System\n\n{SYSTEM}\n\n## User\n\n"
        + "\n".join(b["text"] if b["type"] == "text" else "*[image]*" for b in ex_content),
        encoding="utf-8")

    # Output at effort=high: ~1.5k tokens of JSON plus adaptive thinking,
    # which varies widely; 3k-8k is a working range until measured.
    lo = est_total * PRICE_IN / 1e6 + len(jobs) * 3000 * PRICE_OUT / 1e6
    hi = est_total * PRICE_IN / 1e6 + len(jobs) * 8000 * PRICE_OUT / 1e6
    print(f"~{est_total // max(len(jobs), 1):,} input tokens per sheet; "
          f"estimated ${lo:.2f}–${hi:.2f} for {len(jobs)} sheets ({MODEL}, effort={args.effort})")
    print(f"Example prompt → {example}")
    if args.dry_run:
        return 0

    try:
        import anthropic
    except ImportError:
        sys.exit("Install the 'anthropic' package: pip install anthropic")
    client = anthropic.Anthropic()

    lock = threading.Lock()
    spent = 0.0

    def work(job):
        row, cands, content = job
        rec = call_claude(client, content, args.effort)
        rec.update({"fname": row["fname"], "countryCode": row.get("countryCode", ""),
                    "candidates": cands, "classifier_top1": row["top1_name"],
                    "classifier_prob": float(row["top1_prob"]),
                    "known_answer": row.get("model_class", "")})
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
            open(out_jsonl, "a", encoding="utf-8") as f:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            rec = fut.result()
            flat = flatten(rec)
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                spent += flat["cost_usd"]
            tag = rec.get("error") or f"{flat['best_match']} ({flat['confidence']})" + \
                (" — OUTSIDE CANDIDATES" if flat["outside_candidates"] is True else "")
            print(f"  {rec['fname'][-30:]:<30} classifier: {flat['classifier_top1']:<26} → {tag}")

    recs = [json.loads(l) for l in out_jsonl.read_text(encoding="utf-8").splitlines()]
    flat = pd.DataFrame([flatten(r) for r in recs]).drop_duplicates("fname", keep="last")
    flat.to_csv(args.out / "second_opinion.csv", index=False)
    ok = flat[flat["error"].eq("")]
    print(f"\n${spent:.2f} this run. {len(ok)} sheets assessed in total → second_opinion.csv")
    if len(ok):
        print(f"  outside all candidates: {int(ok['outside_candidates'].eq(True).sum())}")
        print(f"  agrees with classifier top-1: {ok['best_match'].eq(ok['classifier_top1']).mean():.0%}")
        known = ok[ok["known_answer"].ne("")]
        if len(known):
            print(f"  agrees with flora determination: "
                  f"{known['best_match'].eq(known['known_answer']).mean():.0%} (n={len(known)})")
        leak = ok["label_name_seen"].ne("")
        print(f"  a determination was legible on {int(leak.sum())} sheets — check those "
              f"verdicts didn't lean on it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
