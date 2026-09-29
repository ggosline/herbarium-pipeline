"""
Link a trained model and its specimens to the flora wiki.

The wiki (flora-ocr's `wiki/`, one markdown page per taxon, named for the WCVP
accepted name) knows what the flora describes and which sheets its authors
examined. The model knows what sheets look like. Neither knows about the other.
This script joins them, read-only on both sides.

  names      Which model classes have a flora page, and which taxa the flora
             treats that the model can never predict (blind spots). A blind
             spot is a species the model will silently file under some other
             name, so its list belongs beside every set of predictions.

  citations  Parse the specimens the flora cites ("Specimens examined", types)
             into collector + number, match them to GBIF records, and from
             there to specsin and predictions.csv. A sheet the monograph's
             author cited under a name is the nearest thing to an expert-checked
             label this project has. Three things fall out:
               - flora_verified.csv   imaged sheets with a flora determination:
                                      a test set whose labels are not GBIF's
               - indets the flora names   free determinations for 'indet' sheets
               - GBIF name != flora name  candidate mis-determinations — or
                                      re-determinations since the flora; the
                                      review decides which

  rerank     Re-rank each specimen's top-5 by whether its country fits each
             candidate's known range (GBIF records + the flora's countries),
             and suggest blind-spot species the model cannot name: ones the
             flora says resemble a top candidate, or whose own sheets the model
             files under it, and whose range fits. Evaluated against GBIF
             labels and flora_verified.csv at several range weights.

  all        all three (rerank only when --predictions is given).

Matching is collector surname + collection number (with optional letter
suffix). A parse error in a citation just fails to match; a number that matches
a different collector's sheet is rejected by the surname check.

Usage:
  python flora_link.py all \\
      --wiki        E:/flora-ocr/wiki \\
      --nameslist   E:/Diospyros/runs/nameslist.json \\
      --specsin     E:/Diospyros/specsin.csv \\
      --gbif        E:/Diospyros/GBIF-Africa-Feb2025.csv \\
                    E:/Diospyros/GBIFLeiden/occurrence.csv \\
                    E:/Diospyros/GBIFMeise/occurrence.csv \\
                    E:/Diospyros/GBIFParis/occurrence.txt \\
      --predictions E:/Diospyros/review/predictions.csv \\
      --out         E:/Diospyros/review/flora_link
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

csv.field_size_limit(10**8)


# ---------------------------------------------------------------------------
# Wiki pages
# ---------------------------------------------------------------------------

@dataclass
class Page:
    stem: str
    fm: dict
    body: str

    @property
    def name(self) -> str:
        return self.fm.get("name") or self.stem.replace("_", " ")

    @property
    def tags(self) -> list[str]:
        t = self.fm.get("tags", [])
        return t if isinstance(t, list) else [t]

    @property
    def kind(self) -> str:
        # placeholder: accepted for the region, no description ingested
        # novelty:     described in a paper, not in the flora's volumes
        for k in ("placeholder", "novelty"):
            if k in self.tags:
                return k
        return "treated"

    @property
    def countries(self) -> str:
        c = self.fm.get("countries", [])
        return "; ".join(c) if isinstance(c, list) else str(c)


def _flow_list(val: str) -> list[str]:
    return [v.strip().strip("'\"") for v in val.split(",") if v.strip()]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Just enough YAML for the fields used here: scalars, flow lists, and
    block lists. Nested maps (subdivisions, treatments) are skipped."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    fm: dict = {}
    key = None
    for line in text[3:end].splitlines():
        m = re.match(r"^\s*-\s+(.+)$", line)
        if m and key and isinstance(fm.get(key), list):
            fm[key].append(m.group(1).strip().strip("'\""))
            continue
        if not line or line[0] in " \t#":
            continue
        k, sep, val = line.partition(":")
        if not sep:
            continue
        key, val = k.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            fm[key] = _flow_list(val[1:-1])
        elif val == "":
            fm[key] = []          # a block list or a nested map follows
        else:
            fm[key] = val.strip("'\"")
    return fm, text[end + 4:]


def load_pages(wiki: Path, genera: set[str] | None) -> dict[str, Page]:
    pages = {}
    for p in sorted((wiki / "species").glob("*.md")):
        if genera and p.stem.split("_")[0] not in genera:
            continue
        fm, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        pages[p.stem] = Page(p.stem, fm, body)
    return pages


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def fold(s: str) -> str:
    s = str(s).replace("ß", "ss")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z. \-]", " ", fold(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def binomial(s: str) -> str:
    return " ".join(norm_name(s).split()[:2])


class NameIndex:
    """Resolve a name (GBIF, model class, flora) to a wiki page stem.

    Accepted name first, then aliases, then the names the flora used, so a
    synonym that is also some other page's accepted name resolves to that page.
    """

    def __init__(self, pages: dict[str, Page]):
        self.idx: dict[str, tuple[str, str]] = {}
        for via, fld in (("name", "name"), ("alias", "aliases"),
                         ("flora_name", "flora_names")):
            for stem, pg in pages.items():
                vals = pg.fm.get(fld, [])
                if fld == "name":
                    vals = [pg.name]
                for v in vals if isinstance(vals, list) else [vals]:
                    self.idx.setdefault(norm_name(v), (stem, via))

    def resolve(self, name) -> tuple[str | None, str | None]:
        if not isinstance(name, str) or not name.strip():
            return None, None
        n = norm_name(name)
        if n in self.idx:
            return self.idx[n]
        b = binomial(name)
        if b in self.idx:
            return self.idx[b][0], self.idx[b][1] + "(binomial)"
        return None, None

    def suggest(self, name: str) -> str:
        # High cutoff: in one genus, 0.85 already pairs distinct epithets
        # (mafiensis ~ rabiensis). This is for OCR/spelling variants only.
        hit = difflib.get_close_matches(binomial(name), list(self.idx), n=1, cutoff=0.93)
        return self.idx[hit[0]][0] if hit else ""


def load_nameslist(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):  # hierarchical runs: {"species": [...], "genus": [...]}
        data = data.get("species") or next(iter(data.values()))
    return [str(n).replace("_", " ") for n in data]


def load_specsin(path: Path) -> pd.DataFrame:
    sp = pd.read_csv(path, low_memory=False, dtype=str)
    sp["_cat"] = sp["catalogNumber"].map(norm_cat)
    for c in ("indet", "hasfile"):
        if c in sp.columns:
            sp[c] = sp[c].str.lower().eq("true")
    return sp


def norm_cat(c) -> str:
    return re.sub(r"\s+", "", str(c)).upper() if isinstance(c, str) else ""


# ---------------------------------------------------------------------------
# names: model classes <-> flora pages
# ---------------------------------------------------------------------------

def cmd_names(args, pages: dict[str, Page], index: NameIndex) -> None:
    classes = load_nameslist(args.nameslist)
    sp = load_specsin(args.specsin)
    usable = sp[~sp["indet"] & sp.get("hasfile", True)] if "indet" in sp else sp
    n_img = usable["species"].value_counts()

    excluded = set()
    if args.excluded and args.excluded.exists():
        ex = json.loads(args.excluded.read_text(encoding="utf-8"))
        excluded = {str(e.get("species", e) if isinstance(e, dict) else e) for e in
                    (ex if isinstance(ex, list) else ex.get("species", []))}

    rows, page_classes = [], {}
    for c in classes:
        stem, via = index.resolve(c)
        pg = pages.get(stem)
        if stem:
            page_classes.setdefault(stem, []).append(c)
        rows.append({
            "model_class": c,
            "n_images": int(n_img.get(c, 0)),
            "wiki_page": stem or "",
            "wiki_name": pg.name if pg else "",
            "matched_via": via or "",
            "page_kind": pg.kind if pg else "",
            "name_status": pg.fm.get("name_status", "") if pg else "",
            "countries": pg.countries if pg else "",
            "suggestion": "" if stem else index.suggest(c),
        })
    mc = pd.DataFrame(rows)
    # Two classes on one page: the model is being asked to separate synonyms.
    mc["shares_page_with"] = mc.apply(
        lambda r: "; ".join(x for x in page_classes.get(r["wiki_page"], [])
                            if x != r["model_class"]), axis=1)

    # Images per page, under whatever name specsin filed them.
    sp_names = usable["species"].dropna().unique()
    res = {n: index.resolve(n)[0] for n in sp_names}
    page_imgs: dict[str, int] = {}
    page_labels: dict[str, list[str]] = {}
    for n, stem in res.items():
        if stem:
            page_imgs[stem] = page_imgs.get(stem, 0) + int(n_img.get(n, 0))
            page_labels.setdefault(stem, []).append(n)

    blind = []
    for stem, pg in pages.items():
        if stem in page_classes:
            continue
        n = page_imgs.get(stem, 0)
        labels = page_labels.get(stem, [])
        if n == 0:
            reason = "no imaged specimens in training data"
        elif any(l in excluded for l in labels):
            reason = f"dropped from training as too sparse ({n} images)"
        else:
            reason = f"{n} images in specsin, but no model class — sparse or filtered"
        blind.append({
            "wiki_page": stem, "wiki_name": pg.name, "page_kind": pg.kind,
            "name_status": pg.fm.get("name_status", ""),
            "countries": pg.countries, "n_images": n,
            "specsin_names": "; ".join(labels), "reason": reason,
        })
    bs = pd.DataFrame(blind, columns=["wiki_page", "wiki_name", "page_kind", "name_status",
                                      "countries", "n_images", "specsin_names", "reason"])

    args.out.mkdir(parents=True, exist_ok=True)
    mc.to_csv(args.out / "model_classes.csv", index=False)
    bs.to_csv(args.out / "blind_spots.csv", index=False)

    linked = mc["wiki_page"].ne("")
    print(f"\nModel classes: {len(mc)}  with a flora page: {int(linked.sum())}  "
          f"without: {int((~linked).sum())}")
    near = mc[~linked & mc["suggestion"].ne("")]
    if len(near):
        print("  unmatched classes with a near-spelling page (check these by hand):")
        for _, r in near.iterrows():
            print(f"    {r['model_class']:<34} ~ {r['suggestion']}")
    shared = mc[mc["shares_page_with"].ne("")]
    if len(shared):
        print("  classes that are synonyms of each other in the flora:")
        for _, r in shared.iterrows():
            print(f"    {r['model_class']:<34} = {r['shares_page_with']}  [{r['wiki_page']}]")
    print(f"\nBlind spots — flora taxa the model cannot predict: {len(bs)}")
    for kind, g in bs.groupby("page_kind"):
        print(f"  {kind:<12} {len(g):>4}   e.g. {', '.join(g['wiki_name'].head(4))}")
    print(f"\n→ {args.out / 'model_classes.csv'}\n→ {args.out / 'blind_spots.csv'}")


# ---------------------------------------------------------------------------
# citations: flora specimen citations -> GBIF -> specsin -> predictions
# ---------------------------------------------------------------------------

SPEC_HEAD = re.compile(r"(additional |other )?(specimens?|material) (examined|studied|seen)"
                       r"|paratypes?", re.I)
TYPE_HEAD = re.compile(r"^(types?|holotype|lectotype|neotype|syntypes?)\b", re.I)

# FHI 35595, SRFCam, 1267 — an institutional series prefix before the number
CODE = r"(?:[A-Z]{2,}[A-Za-z]*\.?\s*,?\s*)"
NUM = r"(\d{1,6})(?:\s?([a-z])(?![a-zA-Z]))?"
FIRST_NUM = re.compile(rf"(?:^|(?<=[\s(]))({CODE})?{NUM}")
EXTRA_NUM = re.compile(rf"\s*(?:et|and|&)\s+({CODE})?{NUM}")
LIST_NUM = re.compile(r"(?<![\w.])(\d{2,6})(?:\s?([a-z])(?![a-zA-Z]))?(?!\s*(?:km|m|cm|mm)\b)")
# Article style: "... 14 Mar 2019, fr., Bidault et al. 4494 (holotype: MO ...)"
STYLE_B = re.compile(
    r"((?:[A-Z][\w'’\-]+|de|van|der|le|la|du)(?:\s+(?:[A-Z]\.(?:\s?-?[A-Z]\.)*|[A-Z][\w'’\-]+"
    r"|de|van|der|le|la|du|et|&))*?(?:\s+et\s+al\.)?)\s+" + NUM + r"\s*\(")

CONNECT = {"et", "and", "&", "al.", "al", "de", "van", "der", "den", "le", "la", "du",
           "von", "leg.", "coll.", "s.n."}
STOP = {"and", "al", "leg", "coll", "van", "der", "den", "von", "des", "del", "dos",
        "the"}


def clean_md(s: str) -> str:
    s = re.sub(r"\$[^$]*\$", " ", s)           # OCR math residue: $ ^{} $
    s = re.sub(r"[*_`\\]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def looks_like_collector(s: str) -> bool:
    toks = [t for t in re.split(r"[\s,]+", s) if t]
    if not toks or len(toks) > 8 or re.search(r"\d|[:()]", s):
        return False
    if not fold(toks[0])[:1].isupper() and toks[0].lower() not in CONNECT:
        return False
    for t in toks:
        ft = fold(t)
        if t.lower() in CONNECT or re.fullmatch(r"(?:[A-Z]\.-?)+", ft) or ft[:1].isupper():
            continue
        return False
    return True


def surname_keys(s: str) -> frozenset[str]:
    """Distinctive name tokens: no initials, connectors, or series codes."""
    keys = set()
    for t in re.findall(r"[A-Za-z][A-Za-z'\-]*", fold(s)):
        if re.fullmatch(r"[A-Z]{2,}[a-z]*[A-Z]*", t) and not t.istitle():
            continue                           # FHI, SRFCam
        for part in re.split(r"['\-]", t.lower()):
            if len(part) >= 3 and part not in STOP:
                keys.add(part)
    return frozenset(keys)


def initials(s: str) -> set[str]:
    """Single-letter initials: 'De Wilde W.' -> {W}; 'de Wilde J.J.F.E.' -> {J, F, E}."""
    return set(re.findall(r"(?<![A-Za-z])([A-Z])(?![a-z])", fold(s)))


@dataclass
class Citation:
    page: str
    flora_name: str
    role: str
    collector: str
    number: str
    suffix: str = ""
    code: str = ""
    raw: str = ""
    keys: frozenset = field(default_factory=frozenset)


def parse_style_a(par: str) -> list[tuple[str, str, str, str, str]]:
    """Flore du Gabon style: 'Collector 1234, locality (fl., mois) HERB; 5678, ...'
    Returns (collector, code, number, suffix, raw segment)."""
    out, collector = [], None
    for seg in (s.strip() for s in par.split(";")):
        m = FIRST_NUM.search(seg)
        if not m:
            continue
        before = seg[:m.start()].strip(" ,.:")
        if before and not looks_like_collector(before):
            # "échantillons suivants tous de Bipindi, soit ♂ : 1668, 1809, ..."
            if collector and ":" in seg:
                for n in LIST_NUM.finditer(seg.split(":", 1)[1]):
                    out.append((collector, "", n.group(1), n.group(2) or "", seg))
            continue
        if before:
            collector = before
        if not collector:
            continue
        out.append((collector, (m.group(1) or "").strip(" ,."), m.group(2),
                    m.group(3) or "", seg))
        pos = m.end()
        while (x := EXTRA_NUM.match(seg, pos)):   # "5197 et 5211"
            out.append((collector, (x.group(1) or "").strip(" ,."), x.group(2),
                        x.group(3) or "", seg))
            pos = x.end()
    return out


def parse_style_b(text: str) -> list[tuple[str, str, str, str, str]]:
    out = []
    for m in STYLE_B.finditer(text):
        coll = m.group(1).strip()
        if looks_like_collector(coll):
            out.append((coll, "", m.group(2), m.group(3) or "",
                        text[max(0, m.start() - 60):m.end() + 40]))
    return out


def page_citations(pg: Page) -> list[Citation]:
    cits: list[Citation] = []
    flora_name = pg.name
    blocks: list[tuple[str, str, str]] = []    # (role, flora_name, text)

    ts = pg.fm.get("type_specimen")
    if isinstance(ts, str) and ts:
        blocks.append(("type", flora_name, ts))

    role, buf = None, []
    def flush():
        if role and buf:
            blocks.append((role, flora_name, "\n".join(buf)))
    for line in pg.body.splitlines():
        h = re.match(r"^(#{2,4})\s+(.*)$", line)
        if h:
            flush()
            buf = []
            level, head = len(h.group(1)), clean_md(h.group(2))
            if level == 2:
                a = re.match(r"^As (\S+ \S+(?: (?:var|subsp)\. \S+)?)", head)
                flora_name = a.group(1) if a else pg.name
            role = ("examined" if SPEC_HEAD.search(head)
                    else "type" if TYPE_HEAD.match(head) else None)
            continue
        b = re.match(r"^\*\*((?:additional |other )?specimens? examined|paratypes?)\*\*",
                     line, re.I)
        if b and role is None:
            flush()
            role, buf = "examined", []
        if role:
            buf.append(line)
    flush()

    seen = set()
    for role, fname, text in blocks:
        for par in re.split(r"\n\s*\n", text):
            par = clean_md(par)
            if not par:
                continue
            for coll, code, num, suf, raw in parse_style_a(par) + parse_style_b(par):
                keys = surname_keys(coll)
                sig = (keys, num.lstrip("0"), suf)
                if not keys or sig in seen:
                    continue
                seen.add(sig)
                cits.append(Citation(pg.stem, fname, role, coll, num.lstrip("0") or "0",
                                     suf, code, raw[:240], keys))
    return cits


GBIF_KEEP = ("gbifID", "catalogNumber", "institutionCode", "recordedBy", "recordNumber",
             "fieldNumber", "species", "scientificName", "countryCode", "typeStatus",
             "identifiedBy", "dateIdentified", "year")


_GBIF_CACHE: dict[tuple, pd.DataFrame] = {}


def load_gbif(paths: list[Path]) -> pd.DataFrame:
    cache_key = tuple(paths)
    if cache_key in _GBIF_CACHE:
        return _GBIF_CACHE[cache_key]
    frames = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            head = f.readline()
        sep = "\t" if "\t" in head else ","
        df = pd.read_csv(p, sep=sep, dtype=str, usecols=lambda c: c in GBIF_KEEP,
                         quoting=csv.QUOTE_NONE if sep == "\t" else csv.QUOTE_MINIMAL,
                         on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")
        df["gbif_source"] = p.parent.name + "/" + p.name
        frames.append(df)
        print(f"  GBIF {p.name:<28} {len(df):>7,} records")
    g = pd.concat(frames, ignore_index=True).fillna("")
    key = g["gbifID"].where(g["gbifID"].ne(""), g["catalogNumber"])
    g = g[~key.duplicated() | key.eq("")].reset_index(drop=True)
    g["_cat"] = g["catalogNumber"].map(norm_cat)
    g["_name"] = g["species"].where(g["species"].ne(""),
                                    g["scientificName"].map(lambda s: " ".join(s.split()[:2])))

    def num(row):
        s = row["recordNumber"] or row["fieldNumber"]
        hits = list(re.finditer(r"(\d+)(?:\s?([a-z]))?(?![a-z0-9])", s.lower()))
        if not hits:
            return "", ""
        return hits[-1].group(1).lstrip("0") or "0", hits[-1].group(2) or ""
    g[["_num", "_suf"]] = g.apply(lambda r: pd.Series(num(r)), axis=1)
    # Leiden puts collector names in recordNumber too; take tokens from both.
    g["_keys"] = (g["recordedBy"] + " " + g["recordNumber"]).map(surname_keys)
    _GBIF_CACHE[cache_key] = g
    return g


def cmd_citations(args, pages: dict[str, Page], index: NameIndex) -> None:
    classes = load_nameslist(args.nameslist) if args.nameslist else []
    class_of_page: dict[str, str] = {}
    for c in classes:
        stem = index.resolve(c)[0]
        if stem:
            class_of_page.setdefault(stem, c)

    cits = [c for pg in pages.values() for c in page_citations(pg)]
    cdf = pd.DataFrame([{k: v for k, v in c.__dict__.items() if k != "keys"} for c in cits])
    no_cits = [s for s, pg in pages.items()
               if pg.kind != "placeholder" and not any(c.page == s for c in cits)]
    print(f"\nParsed {len(cits):,} citations from {cdf['page'].nunique() if len(cdf) else 0} "
          f"pages ({len(no_cits)} described pages cite no specimens in the wiki)")

    g = load_gbif(args.gbif)
    by_num: dict[str, list[int]] = {}
    for i, n in enumerate(g["_num"]):
        if n:
            by_num.setdefault(n, []).append(i)

    sp = load_specsin(args.specsin)
    sp_by_cat = sp.drop_duplicates("_cat").set_index("_cat")
    pred = None
    if args.predictions and args.predictions.exists():
        pred = pd.read_csv(args.predictions, dtype=str).fillna("")
        pred = pred.drop_duplicates("fname").set_index("fname")

    rows = []
    for ci, c in enumerate(cits):
        for gi in by_num.get(c.number, []):
            gr = g.iloc[gi]
            if c.suffix and gr["_suf"] and c.suffix != gr["_suf"]:
                continue
            if not (c.keys & gr["_keys"]):
                continue
            gname = gr["_name"]
            gstem = index.resolve(gname)[0]
            if not re.match(r"^[A-Z][a-z]+ [a-z]", gname):   # "", "Diospyros L."
                status = "gbif_indet"
            elif gstem == c.page:
                status = "agrees"
            elif gstem is None:
                status = "gbif_name_unresolved"   # e.g. an old synonym the wiki lacks
            else:
                status = "differs"
            # Surname + number can still pair two collectors who share a
            # surname (W.J.J.O. vs J.J.F.E. de Wilde). Flag, don't reject:
            # many citations and GBIF records carry no initials at all.
            fi, gi_ = initials(c.collector), initials(gr["recordedBy"])
            collector_check = "initials differ" if fi and gi_ and not fi & gi_ else ""
            r = {
                "citation": ci, "flora_page": c.page, "flora_name": c.flora_name,
                "flora_accepted": pages[c.page].name, "role": c.role,
                "collector": c.collector, "number": c.number + c.suffix,
                "gbif_recordedBy": gr["recordedBy"], "gbif_recordNumber": gr["recordNumber"] or gr["fieldNumber"],
                "catalogNumber": gr["catalogNumber"], "institutionCode": gr["institutionCode"],
                "gbifID": gr["gbifID"], "gbif_name": gname, "gbif_page": gstem or "",
                "gbif_typeStatus": gr["typeStatus"], "status": status,
                "collector_check": collector_check,
                "model_class": class_of_page.get(c.page, ""),
                "gbif_source": gr["gbif_source"],
            }
            s = sp_by_cat.loc[gr["_cat"]] if gr["_cat"] in sp_by_cat.index else None
            r["fname"] = s["fname"] if s is not None else ""
            r["specsin_species"] = s["species"] if s is not None else ""
            r["specsin_indet"] = bool(s["indet"]) if s is not None else ""
            if pred is not None and r["fname"] in pred.index:
                p = pred.loc[r["fname"]]
                top = [p.get(f"top{k}_name", "") for k in range(1, 6)]
                r["pred_species"] = p["pred_species"]
                r["confidence"] = p["confidence"]
                r["model_correct"] = index.resolve(p["pred_species"])[0] == c.page
                r["in_top5"] = any(index.resolve(t)[0] == c.page for t in top if t)
            rows.append(r)
    m = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(args.out / "citations.csv", index=False)
    m.to_csv(args.out / "citation_matches.csv", index=False)
    if m.empty:
        print("No citations matched a GBIF record.")
        return

    matched_cits = m["citation"].nunique()
    print(f"\nMatched {matched_cits:,} of {len(cits):,} citations "
          f"({matched_cits / max(len(cits), 1):.0%}) to {len(m):,} GBIF records")
    print(m["status"].value_counts().rename("records").to_string())

    flagged = m["collector_check"].ne("")
    if flagged.any():
        print(f"  {int(flagged.sum())} matches flagged 'initials differ' — kept in "
              f"citation_matches.csv, left out of flora_verified.csv")
    ver = m[m["fname"].ne("") & ~flagged].drop_duplicates("fname")
    ver.to_csv(args.out / "flora_verified.csv", index=False)
    print(f"\nFlora-verified imaged sheets (in specsin): {len(ver):,}  → flora_verified.csv")
    if "model_correct" in ver:
        scored = ver[ver["model_correct"].notna() & ver["model_class"].ne("")]
        if len(scored):
            print(f"  model top-1 agrees with flora: {scored['model_correct'].astype(bool).mean():.1%}"
                  f"   top-5: {scored['in_top5'].astype(bool).mean():.1%}   (n={len(scored)};"
                  f" includes training images — optimistic)")
        unpred = ver[ver["model_class"].eq("")]
        if len(unpred):
            print(f"  {len(unpred)} verified sheets are of taxa the model cannot predict")

    indets = ver[ver["specsin_indet"].eq(True)]
    print(f"\nIndets the flora names: {len(indets)}")
    for _, r in indets.head(15).iterrows():
        print(f"  {r['catalogNumber']:<18} {r['collector']} {r['number']:<8} → {r['flora_accepted']}"
              f"   (model: {r.get('pred_species', '')} {r.get('confidence', '')})")

    diff = m[m["status"].eq("differs")].drop_duplicates(["citation", "gbif_name"])
    print(f"\nGBIF name differs from the flora's: {len(diff)} gatherings")
    for _, r in diff.head(25).iterrows():
        print(f"  {r['collector'][:22]:<22} {r['number']:<8} flora: {r['flora_accepted']:<28}"
              f" GBIF: {r['gbif_name']:<28} {r['institutionCode']:<8} {r['collector_check']}")

    write_summary(args.out, pages, cits, no_cits, m, ver, indets, diff)
    print(f"\n→ {args.out}")


def write_summary(out: Path, pages, cits, no_cits, m, ver, indets, diff) -> None:
    per_page = (m.drop_duplicates("citation").groupby("flora_page").size()
                .rename("matched").to_frame()
                .join(pd.Series([c.page for c in cits]).value_counts().rename("cited"), how="right")
                .fillna(0).astype(int).sort_values("cited", ascending=False))
    lines = [
        "# Flora ↔ specimen link",
        "",
        f"- wiki pages in scope: {len(pages)}",
        f"- specimen citations parsed: {len(cits)}  (matched to GBIF: {m['citation'].nunique()})",
        f"- flora-verified imaged sheets: {len(ver)}",
        f"- indets the flora names: {len(indets)}",
        f"- gatherings where GBIF's name differs from the flora's: {len(diff)}",
        "",
        "## Described pages with no specimen citations in the wiki",
        "",
        ", ".join(no_cits) or "none",
        "",
        "## Citations matched per page",
        "",
        "| page | cited | matched |", "|---|---|---|",
        *[f"| {s} | {r.cited} | {r.matched} |" for s, r in per_page.iterrows()],
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# rerank: range evidence + blind-spot candidates
# ---------------------------------------------------------------------------

COUNTRY_ISO = {
    "nigeria": "NG", "cameroon": "CM", "equatorial guinea": "GQ", "bioko": "GQ",
    "sao tome and principe": "ST", "sao tome": "ST", "gabon": "GA",
    "republic of the congo": "CG", "congo": "CG", "congo-brazzaville": "CG",
    "democratic republic of the congo": "CD", "congo-kinshasa": "CD", "zaire": "CD",
    "angola": "AO", "cabinda": "AO", "central african republic": "CF",
    "benin": "BJ", "togo": "TG", "ghana": "GH", "cote d'ivoire": "CI",
    "ivory coast": "CI", "liberia": "LR", "sierra leone": "SL", "guinea": "GN",
    "guinea-bissau": "GW", "senegal": "SN", "gambia": "GM", "mali": "ML",
    "burkina faso": "BF", "niger": "NE", "chad": "TD", "sudan": "SD",
    "south sudan": "SS", "ethiopia": "ET", "eritrea": "ER", "somalia": "SO",
    "djibouti": "DJ", "kenya": "KE", "uganda": "UG", "tanzania": "TZ", "rwanda": "RW",
    "burundi": "BI", "zambia": "ZM", "malawi": "MW", "mozambique": "MZ",
    "zimbabwe": "ZW", "botswana": "BW", "namibia": "NA", "south africa": "ZA",
    "eswatini": "SZ", "swaziland": "SZ", "lesotho": "LS", "madagascar": "MG",
    "comoros": "KM", "mauritius": "MU", "seychelles": "SC", "mauritania": "MR",
}
# Coarse phytochoria. Collecting is patchy, so a species recorded in Gabon
# should not be penalised for a gap in Equatorial Guinea: range evidence is
# half country, half region.
REGIONS = {
    "upper_guinea": "SN GM GW GN SL LR CI GH TG BJ",
    "lower_guinea": "NG CM GQ ST GA CG",
    "congolian":    "CD CF",
    "sudano_sahel": "MR ML BF NE TD SD SS",
    "east":         "ET ER DJ SO KE UG TZ RW BI",
    "zambezian":    "AO ZM MW MZ ZW NA BW",
    "southern":     "ZA SZ LS",
    "madagascar":   "MG KM MU SC",
}
REGION_OF = {c: r for r, cs in REGIONS.items() for c in cs.split()}
WIKI_PSEUDO = 2    # a country the flora lists counts as this many records
ALPHA = 0.5        # additive smoothing: an unrecorded country is unlikely, not impossible
NO_RANGE_SECTIONS = re.compile(r"specimens|synonym|see also|^types?$|figures|source", re.I)


def country_iso(name: str) -> str:
    n = re.sub(r"^the ", "", fold(name).lower().strip())
    return COUNTRY_ISO.get(n, "")


class Ranges:
    """Record counts per taxon per country, and the likelihood of a country
    under each taxon's range."""

    def __init__(self):
        self.n: dict[str, dict[str, float]] = {}

    def add(self, taxon: str, iso: str, k: float = 1) -> None:
        if taxon and iso in REGION_OF:
            d = self.n.setdefault(taxon, {})
            d[iso] = d.get(iso, 0) + k

    def regions(self, taxon: str) -> set[str]:
        return {REGION_OF[c] for c in self.n.get(taxon, {})}

    def fit(self, taxon: str, iso: str, drop: float = 0) -> float:
        """Likelihood of `iso` under `taxon`'s range, relative to a taxon with no
        range data (1.0 = no evidence either way). `drop` removes the
        specimen's own record so its label cannot vote for itself."""
        if iso not in REGION_OF:
            return 1.0
        return self._lik(self.n.get(taxon, {}), iso, drop) / self._lik({}, iso, 0)

    @staticmethod
    def _lik(cnt: dict, iso: str, drop: float) -> float:
        n_s = max(sum(cnt.values()) - drop, 0)
        n_c = max(cnt.get(iso, 0) - drop, 0)
        reg = REGION_OF[iso]
        n_r = max(sum(v for c, v in cnt.items() if REGION_OF.get(c) == reg) - drop, 0)
        p_c = (n_c + ALPHA) / (n_s + ALPHA * len(REGION_OF))
        p_r = (n_r + ALPHA) / (n_s + ALPHA * len(REGIONS)) / len(REGIONS[reg].split())
        return 0.5 * p_c + 0.5 * p_r


def related_names(pg: Page) -> set[str]:
    """Species a page's prose compares it with: wiki links and 'D. xxx' mentions
    outside the specimen, synonymy and figure sections."""
    genus = pg.stem.split("_")[0]
    keep, skip = [], False
    for line in pg.body.splitlines():
        h = re.match(r"^#{2,4}\s+(.*)$", line)
        if h:
            skip = bool(NO_RANGE_SECTIONS.search(clean_md(h.group(1))))
            continue
        if not skip and not line.startswith("**Nomenclature**"):
            keep.append(line)
    text = "\n".join(keep)
    names = {s.replace("_", " ") for s in
             re.findall(r"\[\[([A-Z][a-z]+_[a-z][a-z\-]+)(?:\|[^\]]*)?\]\]", text)}
    names |= {f"{genus} {e}" for e in
              re.findall(rf"\b(?:{genus}|{genus[0]}\.)\s+([a-z][a-z\-]{{2,}})", clean_md(text))}
    return names


def cmd_rerank(args, pages: dict[str, Page], index: NameIndex) -> None:
    classes = load_nameslist(args.nameslist)
    cls = set(classes)
    class_of_page: dict[str, str] = {}
    for c in classes:
        stem = index.resolve(c)[0]
        if stem:
            class_of_page.setdefault(stem, c)

    def tkey(name) -> str:
        """One key per taxon: the model class if it has one, else the page."""
        if not isinstance(name, str) or not name.strip():
            return ""
        if name in cls:
            return name
        stem = index.resolve(name)[0]
        if stem:
            return class_of_page.get(stem, "page:" + stem)
        return " ".join(name.split()[:2])

    # --- ranges -----------------------------------------------------------
    g = load_gbif(args.gbif)
    g = g[g["_name"].str.match(r"^[A-Z][a-z]+ [a-z]")]
    g_key = g["_name"].map(tkey)
    g_iso = g["countryCode"].str.upper().str.strip()
    ranges = Ranges()
    for t, iso in zip(g_key, g_iso):
        ranges.add(t, iso)
    n_wiki = 0
    for stem, pg in pages.items():
        for c in (pg.fm.get("countries") or []):
            iso = country_iso(c)
            if iso:
                ranges.add(tkey(pg.name), iso, WIKI_PSEUDO)
                n_wiki += 1
    print(f"\nRanges: {len(g):,} GBIF records + {n_wiki} flora country listings "
          f"over {len(ranges.n)} taxa")

    # --- specimens --------------------------------------------------------
    sp = load_specsin(args.specsin)
    sp_by_fname = sp.drop_duplicates("fname").set_index("fname")
    g_by_cat: dict[str, list[tuple[str, str]]] = {}
    for cat, t, iso in zip(g["_cat"], g_key, g_iso):
        if cat:
            g_by_cat.setdefault(cat, []).append((t, iso))

    pred = pd.read_csv(args.predictions, dtype=str).fillna("")
    pred["_indet"] = pred["indet"].str.lower().eq("true")
    K = [k for k in range(1, 6) if f"top{k}_name" in pred.columns]

    iso_of, own_of = [], []
    for f in pred["fname"]:
        s = sp_by_fname.loc[f] if f in sp_by_fname.index else None
        cat = s["_cat"] if s is not None else ""
        own = g_by_cat.get(cat, [])
        iso = str(s["countryCode"]).upper().strip() if s is not None and isinstance(s["countryCode"], str) else ""
        if not iso and own:
            iso = own[0][1]
        iso_of.append(iso)
        own_of.append(own)
    pred["countryCode"] = iso_of

    names = pred[[f"top{k}_name" for k in K]].to_numpy()
    probs = pred[[f"top{k}_prob" for k in K]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    keys = [[tkey(n) for n in row] for row in names]
    fits = []
    for row_keys, iso, own in zip(keys, iso_of, own_of):
        fits.append([ranges.fit(t, iso, drop=sum(1 for ot, oi in own if ot == t and oi == iso))
                     for t in row_keys])
    fits = pd.DataFrame(fits).to_numpy()

    def rerank(w: float):
        post = probs * fits ** w
        tot = post.sum(axis=1, keepdims=True)
        post = post / pd.DataFrame(tot).replace(0, 1).to_numpy() * probs.sum(axis=1, keepdims=True)
        order = (-post).argsort(axis=1, kind="stable")
        return post, order

    # --- evaluation -------------------------------------------------------
    label = pred["true_species"].map(tkey)
    labelled = ~pred["_indet"] & label.isin(cls) & pred["countryCode"].ne("")
    conf = pd.to_numeric(pred["confidence"], errors="coerce").fillna(0)
    ver_label = pd.Series("", index=pred.index)
    vpath = args.out / "flora_verified.csv"
    if vpath.exists():
        v = pd.read_csv(vpath, dtype=str).fillna("")
        v = v[v["model_class"].ne("")].set_index("fname")["model_class"]
        ver_label = pred["fname"].map(v).fillna("")
    verified = ver_label.ne("") & pred["countryCode"].ne("")

    top1_key = lambda order: [keys[i][order[i, 0]] for i in range(len(keys))]
    base = pd.Series(top1_key(rerank(0)[1]), index=pred.index)
    print(f"\nTop-1 accuracy by range weight w (w=0 is the model alone)")
    print(f"  {'w':>4}  {'GBIF labels':>12}  {'  conf<0.5':>10}  {'flora-verified':>15}"
          f"  {'fixed':>6} {'broken':>6}")
    for w in args.weights:
        t1 = pd.Series(top1_key(rerank(w)[1]), index=pred.index)
        ok, ok0 = t1.eq(label), base.eq(label)
        low = labelled & (conf < 0.5)
        vacc = t1[verified].eq(ver_label[verified]).mean() if verified.any() else float("nan")
        fixed = int((ok & ~ok0 & labelled).sum())
        broken = int((~ok & ok0 & labelled).sum())
        print(f"  {w:>4}  {ok[labelled].mean():>12.1%}  {ok[low].mean():>10.1%}  {vacc:>15.1%}"
              f"  {fixed:>6} {broken:>6}")
    print(f"  (n = {int(labelled.sum()):,} labelled, {int((labelled & (conf < 0.5)).sum()):,} "
          f"with conf<0.5, {int(verified.sum())} flora-verified. Labels are mostly training "
          f"images; GBIF ranges come from the same records, own record left out.)")

    # --- blind-spot candidates --------------------------------------------
    blind = {s: pg for s, pg in pages.items() if s not in class_of_page}
    triggers: dict[str, dict[str, str]] = {}      # blind stem -> {class: why}
    for stem, pg in blind.items():
        trig = {}
        for n in related_names(pg):
            t = tkey(n)
            if t in cls and t != tkey(pg.name):
                trig[t] = "flora compares it with"
        # Classes the model files this taxon's own imaged sheets under. One
        # stray sheet is noise, not a pattern: need 2+ and a quarter of them.
        own = pred[pred["true_species"].map(tkey).eq(tkey(pg.name))]
        for t, n in own["top1_name"].map(tkey).value_counts().items():
            if t in cls and n >= max(2, 0.25 * len(own)):
                trig.setdefault(t, f"model files {n}/{len(own)} of its sheets as")
        triggers[stem] = trig

    blind_col = []
    for i, (row_keys, iso) in enumerate(zip(keys, iso_of)):
        hits = []
        for stem, trig in triggers.items():
            via = [t for t in row_keys[:2] if t in trig]
            if not via:
                continue
            # Country, not region: these are mostly narrow endemics, and a
            # regional match would put D. onanae on every Lower Guinea sheet.
            recorded = ranges.n.get(tkey(blind[stem].name), {})
            if iso and recorded and iso not in recorded:
                continue
            fit = f"recorded in {iso}" if recorded and iso else "range unknown"
            hits.append(f"{blind[stem].name} ({trig[via[0]]} {via[0]}; {fit})")
        blind_col.append(" | ".join(hits))

    # --- write ------------------------------------------------------------
    post, order = rerank(args.geo_weight)
    out = pred[["fname", "countryCode", "true_species"]].copy()
    out["indet"] = pred["_indet"]
    out["orig_top1"] = pred["top1_name"]
    out["orig_conf"] = conf
    for r in range(len(K)):
        idx = order[:, r]
        rows = range(len(pred))
        out[f"top{r + 1}_name"] = [names[i, idx[i]] for i in rows]
        out[f"top{r + 1}_prob"] = [round(float(post[i, idx[i]]), 4) for i in rows]
        out[f"top{r + 1}_range_fit"] = [round(float(fits[i, idx[i]]), 2) for i in rows]
    out["changed"] = out["top1_name"].ne(out["orig_top1"])
    out["blind_spot_candidates"] = blind_col
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "reranked.csv", index=False)

    ind = out[out["indet"]]
    print(f"\nAt w={args.geo_weight}: top-1 changes on {int(out['changed'].sum()):,} of "
          f"{len(out):,} specimens ({int(ind['changed'].sum())} of {len(ind)} indets)")
    for _, r in ind[ind["changed"]].sort_values("orig_conf").head(12).iterrows():
        print(f"  {r['fname'][-26:]:<26} {r['countryCode']:<3} {r['orig_top1']:<26} "
              f"{r['orig_conf']:.2f} → {r['top1_name']:<26} (range ×{r['top1_range_fit']})")
    print(f"\nBlind-spot candidates: {int(out['blind_spot_candidates'].ne('').sum()):,} specimens "
          f"({int(ind['blind_spot_candidates'].ne('').sum())} indets)")
    for stem, trig in triggers.items():
        n = int(out["blind_spot_candidates"].str.contains(re.escape(blind[stem].name)).sum())
        why = "; ".join(f"{v} {k}" for k, v in trig.items()) or "no trigger (nothing links it to a class)"
        print(f"  {blind[stem].name:<26} {n:>5}   {why}")
    print(f"\n→ {args.out / 'reranked.csv'}")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["names", "citations", "rerank", "all"])
    p.add_argument("--wiki", type=Path, required=True, help="flora-ocr wiki/ directory.")
    p.add_argument("--nameslist", type=Path, help="Model's nameslist.json.")
    p.add_argument("--specsin", type=Path, required=True)
    p.add_argument("--gbif", type=Path, nargs="*", default=[],
                   help="GBIF occurrence files (csv or DwC-A occurrence.txt) with "
                        "recordedBy/recordNumber. Needed for 'citations'.")
    p.add_argument("--predictions", type=Path, help="predictions.csv from identify_herbarium.")
    p.add_argument("--excluded", type=Path, help="excluded_species.json (optional).")
    p.add_argument("--genus", nargs="*",
                   help="Genera in scope (default: the genera of the model's classes).")
    p.add_argument("--geo-weight", type=float, default=0.5,
                   help="rerank: exponent on the range evidence (0 = model alone). "
                        "0.5 was best on Diospyros; w=1 over-trusts GBIF ranges.")
    p.add_argument("--weights", type=float, nargs="*", default=[0, 0.25, 0.5, 0.75, 1],
                   help="rerank: range weights to evaluate.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.command in ("names", "rerank", "all") and not args.nameslist:
        p.error(f"{args.command} needs --nameslist")
    if args.command in ("citations", "rerank", "all") and not args.gbif:
        p.error(f"{args.command} needs --gbif")
    if args.command == "rerank" and not args.predictions:
        p.error("rerank needs --predictions")
    if args.excluded is None and args.nameslist:
        args.excluded = args.nameslist.parent / "excluded_species.json"

    genera = set(args.genus or [])
    if not genera and args.nameslist:
        genera = {n.split()[0] for n in load_nameslist(args.nameslist)}
    pages = load_pages(args.wiki, genera or None)
    print(f"Wiki: {len(pages)} species pages in {', '.join(sorted(genera)) or 'all genera'}")
    # Index over the whole wiki, so a name that moved genus still resolves.
    index = NameIndex(load_pages(args.wiki, None) if genera else pages)

    if args.command in ("names", "all"):
        cmd_names(args, pages, index)
    if args.command in ("citations", "all"):
        cmd_citations(args, pages, index)
    if args.command == "rerank" or (args.command == "all" and args.predictions):
        cmd_rerank(args, pages, index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
