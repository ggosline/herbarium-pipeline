"""Step 5 of the interpretability study: does the model encode botanical traits?

Concept activation vectors (Kim et al. 2018, TCAV). A concept — 'elongate
leaves', 'opposite phyllotaxy', 'entire margin' — is defined by example
specimens that have it and example specimens that do not. A linear classifier
separating their activations gives a CAV: a direction in activation space that
means the concept. The directional derivative of a class logit along that CAV
says whether the concept pushes the model TOWARDS that taxon.

  from-treatments— parse published taxonomic treatments (TaxPub XML, e.g. the
                   PhytoKeys Flora of Cameroon Annonaceae) into species-level
                   trait concepts: secondary vein pairs, blade size and shape,
                   habit, blade indumentum. This is where the real botanical
                   concepts come from — descriptions already written by
                   taxonomists, instead of scoring sheets by hand.
  auto-concepts  — derive proxy concepts (specimen bulk, elongation) from the
                   PC1 plant masks. No botanist required; these exist to prove
                   the pipeline end to end and to calibrate what a real effect
                   looks like.
  template       — write a CSV and contact sheets for scoring real traits by
                   hand. This is the step that needs a botanist.
  cav            — train CAVs from a labels CSV and report TCAV scores per
                   taxon, against shuffled-label CAVs as the null.

Why not the last layer: the classifier head is linear, so the directional
derivative there is the same for every specimen and the TCAV score collapses to
0 or 1. CAVs are taken at an intermediate block (--layer, default 6 from the
end) where the remaining network is still nonlinear.

Usage:
  python probe_concepts.py auto-concepts --out DIR
  python probe_concepts.py cav      --checkpoint CKPT --out DIR --labels auto_concepts.csv
  python probe_concepts.py template --out DIR --n 120
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import probe_embeddings as pe
from identify_herbarium import IMAGENET_MEAN, IMAGENET_STD, encode_coords
from probe_confounds import build_full_model
from probe_embeddings import META_NAME, model_view

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Concepts from published treatments
# ---------------------------------------------------------------------------

TAXPUB_NS = {"tp": "http://www.plazi.org/taxpub"}

# Species-level traits are a compromise worth stating plainly: the treatment
# describes the TAXON, this sheet shows one specimen of it, and the character
# may not be visible on this particular sheet (no fruit, a juvenile shoot).
# That is label noise, and it biases TCAV towards zero rather than inventing an
# effect — but it also means a CAV can learn 'these species' instead of the
# trait, which is what the cross-genus control in `cav` exists to catch.
# Every revision words its descriptions differently, so each trait is a LIST of
# alternatives tried in order. Observed so far:
#   Cameroon flora   "secondary veins 8 to 12 pairs"  / "blade 7-28 cm long, 2-9.5 cm wide"
#   Xylopia monograph"secondary veins ..., 7-11 per side" / "blades 5.1-11.8 cm long, ..."
#   Monanthotaxis    "secondary veins ... 7-9 per side"  / "lamina ... 8.4-14 by 2.9-5.2 cm"
# A pattern fitted to one work extracts NOTHING from another while habit and
# indumentum keep parsing, so the output looks healthy — check fill rates.
VEIN_PATTERNS = [
    re.compile(r"secondary veins?[^.]{0,100}?(\d+)\s*(?:to|[-–])\s*(\d+)"
               r"\s*(?:pairs|per side)", re.I),
]
BLADE_PATTERNS = [
    re.compile(r"blades?[^.]{0,80}?([\d.]+)\s*[-–]\s*([\d.]+)\s*cm long,\s*"
               r"([\d.]+)\s*[-–]\s*([\d.]+)\s*cm wide", re.I),
    re.compile(r"lamina[^.]{0,200}?([\d.]+)\s*[-–]\s*([\d.]+)\s*by\s*"
               r"([\d.]+)\s*[-–]\s*([\d.]+)\s*cm", re.I),
]
RATIO_PATTERN = re.compile(r"([\d.]+)\s*[-–]\s*([\d.]+)\s*times longer than wide", re.I)
LIANA_PATTERN = re.compile(r"\b(liana|climb\w*|scandent|twining|lianescent)", re.I)
TREE_PATTERN = re.compile(r"\b(tree|shrub)", re.I)


def extract_traits(text: str) -> dict:
    """Measured traits from one species description, whatever its house style."""
    row: dict = {}
    for pat in VEIN_PATTERNS:
        m = pat.search(text)
        if m:
            row["vein_pairs"] = (int(m.group(1)) + int(m.group(2))) / 2
            break
    for pat in BLADE_PATTERNS:
        m = pat.search(text)
        if m:
            lo_l, hi_l, lo_w, hi_w = (float(m.group(i)) for i in range(1, 5))
            row["blade_len"] = (lo_l + hi_l) / 2
            row["blade_wid"] = (lo_w + hi_w) / 2
            row["blade_ratio"] = row["blade_len"] / max(row["blade_wid"], 1e-6)
            break
    # Some works state elongation outright, which beats dividing two midpoints.
    m = RATIO_PATTERN.search(text)
    if m:
        row["blade_ratio"] = (float(m.group(1)) + float(m.group(2))) / 2
    # Habit comes from the sentence that opens the description ("Shrub or
    # liana, to 6 m long; ..."), located by its opening word rather than by a
    # fixed offset: in a PDF revision the synonymy and type block sit between
    # the heading and the description, so any window measured from the start
    # either misses the habit or drags in unrelated prose.
    m = re.search(r"(?:^|[.;]\s)((?:Trees?|Shrubs?|Lianas?|Climbers?|Herbs?)"
                  r"[^.]{0,300})", text)
    habit_sentence = m.group(1) if m else text[:220]
    if LIANA_PATTERN.search(habit_sentence):
        row["habit_liana"] = 1
    elif TREE_PATTERN.search(habit_sentence):
        row["habit_liana"] = 0
    # Blade indumentum specifically, not the whole plant's: almost every
    # description says both 'glabrous' and 'pubescent' about some organ.
    seg = re.split(r"secondary veins?", text, flags=re.I)[0]
    low = seg.lower()
    cut = max(low.find("blade"), low.find("lamina"))
    seg = seg[cut:] if cut >= 0 else ""
    if seg:
        pub = len(re.findall(r"pubescen|puberul|tomentos|villous|hairy|sericeous|hairs",
                             seg, re.I))
        gla = len(re.findall(r"glabrous|glabrate", seg, re.I))
        if pub or gla:
            row["blade_hairy"] = int(pub > gla)
    return row


def parse_pdf_treatments(path: Path) -> pd.DataFrame:
    """Species descriptions from a plain PDF revision with a text layer.

    Treatments are found by their numbered headings ("4. Monanthotaxis
    atopostema"). Where a species appears more than once (key entries, running
    heads) the longest block wins, which is the description rather than the
    cross-reference.
    """
    import fitz

    doc = fitz.open(str(path))
    text = "\n".join(page.get_text() for page in doc)
    text = text.replace("\u00ad", "")                      # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)               # hyphenated line breaks
    # Headings must be matched BEFORE newlines are collapsed: only the line
    # anchor separates a real treatment heading from a literature citation like
    # "... (1971b) 30, non Monanthotaxis angustifolia". Without it the parser
    # finds three times too many "species", each with the wrong text attached.
    heads = list(re.finditer(r"^\s{0,4}(\d+)\.\s+([A-Z][a-z]+)\s+([a-z\-]{3,})\b",
                             text, re.M))
    if not heads:
        raise SystemExit(f"ERROR: no numbered species headings found in {path.name}.")
    best: dict[str, str] = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        name = f"{h.group(2)} {h.group(3)}"
        seg = re.sub(r"\s*\n\s*", " ", text[h.start():end])
        if len(seg) > len(best.get(name, "")):
            best[name] = seg
    rows = []
    for name, seg in best.items():
        row = {"species": name, "source": path.name}
        row.update(extract_traits(seg))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_treatments(paths: list[Path]) -> pd.DataFrame:
    """Species -> measured traits, from TaxPub taxonomic treatment XML."""
    from lxml import etree

    rows = []
    for path in paths:
        root = etree.parse(str(path)).getroot()
        for tx in root.findall(".//tp:taxon-treatment", TAXPUB_NS):
            nm = tx.find(".//tp:taxon-name", TAXPUB_NS)
            if nm is None:
                continue
            parts = {q.get("taxon-name-part-type"): (q.text or "").strip()
                     for q in nm.findall(".//tp:taxon-name-part", TAXPUB_NS)}
            genus, epithet = parts.get("genus"), parts.get("species")
            if not (genus and epithet):
                continue
            secs = [q for q in tx.findall(".//tp:treatment-sec", TAXPUB_NS)
                    if q.get("sec-type") == "description"]
            if not secs:
                continue
            text = " ".join("".join(q.itertext()) for q in secs[0].findall(".//p"))
            row = {"species": f"{genus} {epithet}", "source": path.name}
            row.update(extract_traits(text))
            rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("species")


def stage_from_treatments(args) -> None:
    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    frames = []
    if args.xml:
        frames.append(parse_treatments([Path(p) for p in args.xml]))
    for pdf in (args.pdf or []):
        frames.append(parse_pdf_treatments(Path(pdf)))
    tr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not tr.empty:
        # Prefer the row that actually carries measurements when a species is
        # treated in more than one work.
        tr["_filled"] = tr.notna().sum(axis=1)
        tr = (tr.sort_values("_filled", ascending=False)
                .drop_duplicates("species").drop(columns="_filled"))
    if tr.empty:
        raise SystemExit("ERROR: no treatments parsed — is this TaxPub XML?")
    ours = set(df["species"].unique())
    tr = tr[tr.species.isin(ours)]
    covered = df.species.isin(set(tr.species)).sum()
    print(f"  {len(tr)} of {len(ours)} held-out species matched, "
          f"covering {covered:,} of {len(df):,} specimens")

    rows = []
    for name, col, kind in [("many_veins", "vein_pairs", "tercile"),
                            ("large_blade", "blade_len", "tercile"),
                            ("elongate_blade", "blade_ratio", "tercile"),
                            ("habit_liana", "habit_liana", "binary"),
                            ("blade_hairy", "blade_hairy", "binary")]:
        if col not in tr.columns:
            continue
        vals = tr[["species", col]].dropna()
        if kind == "tercile":
            lo, hi = np.percentile(vals[col], [33, 67])
            lab = vals.assign(value=np.where(vals[col] >= hi, 1,
                                             np.where(vals[col] <= lo, 0, -1)))
        else:
            lab = vals.assign(value=vals[col].astype(int))
        lab = lab[lab.value >= 0]
        merged = df.merge(lab[["species", "value"]], on="species")
        if merged.value.nunique() < 2 or len(merged) < args.min_specimens:
            print(f"  {name}: skipped ({len(merged)} specimens, "
                  f"{merged.value.nunique()} classes)")
            continue
        rows.append(pd.DataFrame({"fname": merged["fname"], "concept": name,
                                  "value": merged["value"]}))
        n_sp = merged.species.nunique()
        n_gen = merged.genus.nunique()
        print(f"  {name}: {int((merged.value == 1).sum())} positive / "
              f"{int((merged.value == 0).sum())} negative specimens, "
              f"{n_sp} species across {n_gen} genera")
    if not rows:
        raise SystemExit("ERROR: no usable concepts.")
    dest = out / "treatment_concepts.csv"
    pd.concat(rows).to_csv(dest, index=False)
    print(f"  Wrote {dest}")


# ---------------------------------------------------------------------------
# Proxy concepts from the plant masks
# ---------------------------------------------------------------------------

def stage_auto_concepts(args) -> None:
    """Two concepts derivable without a botanist, from the PC1 masks.

    'bulk'      — how much sheet the specimen covers (a big leafy branch vs a
                  sparse twig).
    'elongate'  — the mask's second-moment ratio: narrow, strap-like material
                  versus broad and round.

    Crude on purpose. They are here so the CAV machinery can be validated on
    something with a known image basis before a botanist spends hours scoring
    real traits.
    """
    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    mask_file = out / "plant_masks.npz"
    if not mask_file.exists():
        raise SystemExit("ERROR: plant_masks.npz missing — run `probe_confounds.py mask`.")
    masks = np.load(mask_file)["masks"]
    if len(masks) != len(df):
        raise SystemExit(f"ERROR: {len(masks)} masks for {len(df)} specimens — "
                         "the cache and the mask file are out of step.")

    bulk = masks.mean(axis=(1, 2))
    elong = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(ys) < 10:
            elong.append(np.nan)
            continue
        cov = np.cov(np.stack([ys - ys.mean(), xs - xs.mean()]))
        w = np.linalg.eigvalsh(cov)
        elong.append(float(np.sqrt(max(w[1], 1e-9) / max(w[0], 1e-9))))
    elong = np.array(elong)

    # Terciles, middle discarded: a concept needs clear positives and clear
    # negatives, and the ambiguous middle is what makes a CAV meaningless.
    rows = []
    for name, vals in (("bulk", bulk), ("elongate", elong)):
        ok = np.isfinite(vals)
        lo, hi = np.nanpercentile(vals[ok], [33, 67])
        label = np.where(vals >= hi, 1, np.where(vals <= lo, 0, -1))
        label[~ok] = -1
        rows.append(pd.DataFrame({"fname": df["fname"], "concept": name,
                                  "value": label, "raw": np.round(vals, 4)}))
        kept = (label >= 0).sum()
        print(f"  {name}: {(label == 1).sum()} positive, {(label == 0).sum()} negative, "
              f"{len(df) - kept} discarded as ambiguous "
              f"(cut at {lo:.3f} / {hi:.3f})")
    lab = pd.concat(rows)
    lab = lab[lab.value >= 0]
    dest = out / "auto_concepts.csv"
    lab.to_csv(dest, index=False)
    print(f"  Wrote {dest}")


# ---------------------------------------------------------------------------
# Labelling template for real traits
# ---------------------------------------------------------------------------

TRAITS = ["margin_entire", "phyllotaxy_alternate", "internode_long",
          "leaf_elongate", "indumentum_present", "fruit_present"]


def stage_template(args) -> None:
    """Contact sheets + a CSV for a botanist to score, one row per specimen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    df = pd.read_csv(out / META_NAME)
    rng = np.random.RandomState(args.seed)
    # Stratify by genus so the sheets span the family rather than whatever the
    # sort order happens to put first.
    per_genus = max(1, args.n // max(df["genus"].nunique(), 1))
    sel = (df.groupby("genus", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), per_genus), random_state=args.seed),
                    include_groups=False)
             .reset_index(drop=True))
    if len(sel) > args.n:
        sel = sel.iloc[rng.permutation(len(sel))[:args.n]]
    sel = sel.sort_values(["genus", "species"]).reset_index(drop=True)

    tmpl = pd.DataFrame({"fname": sel["fname"], "species": sel["species"],
                         "genus": sel["genus"], "sheet": "", "position": ""})
    for t in args.traits.split(","):
        tmpl[t] = ""            # 1 = has it, 0 = clearly lacks it, blank = unsure
    per_page = args.per_page
    for page in range(int(np.ceil(len(sel) / per_page))):
        chunk = sel.iloc[page * per_page:(page + 1) * per_page]
        ncol = 4
        nrow = int(np.ceil(len(chunk) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3.4))
        axes = np.atleast_2d(axes)
        for i, (_, row) in enumerate(chunk.reset_index(drop=True).iterrows()):
            ax = axes[i // ncol][i % ncol]
            ax.imshow(model_view(row["path"], 512))
            ax.set_title(f"{page + 1}.{i + 1}  {row['species']}", fontsize=6)
        for ax in axes.ravel():
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"concept_sheet_{page + 1:02d}.png", dpi=args.dpi)
        plt.close(fig)
        idx = tmpl.index[(page * per_page):(page + 1) * per_page]
        tmpl.loc[idx, "sheet"] = page + 1
        tmpl.loc[idx, "position"] = [f"{page + 1}.{i + 1}" for i in range(len(idx))]
    dest = out / "concept_labels_TEMPLATE.csv"
    tmpl.to_csv(dest, index=False)
    print(f"  Wrote {dest} and {int(np.ceil(len(sel) / per_page))} contact sheets.")
    print(f"  Score each trait 1 (present), 0 (clearly absent), blank (unsure — "
          f"blank rows are dropped, which is the right thing to do with a doubtful call).")
    print(f"  Then: python probe_concepts.py cav --labels {dest.name} --long-format false")


# ---------------------------------------------------------------------------
# CAVs and TCAV
# ---------------------------------------------------------------------------

def layer_module(backbone, layer: int):
    blocks = backbone.blocks
    idx = layer if layer >= 0 else len(blocks) + layer
    if not 0 <= idx < len(blocks):
        raise SystemExit(f"ERROR: --layer {layer} out of range for {len(blocks)} blocks.")
    return blocks[idx], idx


def activations(model, backbone, paths, geo, layer, args, device, want_grad_for=None):
    """Pooled patch-token activations at one block, and optionally d(logit)/d(act).

    The gradient is what makes this TCAV rather than a plain probe: it says
    whether moving along the concept direction raises the class logit, not
    merely whether the concept is decodable.
    """
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    blk, _ = layer_module(backbone, layer)
    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    store: dict = {}

    def hook(_m, _i, out):
        store["act"] = out
        out.retain_grad()

    # Keeping the full backward graph for 24 blocks x 1605 tokens OOMs a 24 GB
    # card at batch 8 (measured: 13.3 GiB at batch 4, OOM at 8). Gradient
    # checkpointing plus bf16 brings batch 8 down to 3.6 GiB by recomputing
    # activations instead of storing them — the recompute is cheap next to the
    # forward pass itself.
    was_ckpt = getattr(backbone, "grad_checkpointing", False)
    backbone.set_grad_checkpointing(True)
    handle = blk.register_forward_hook(hook)
    acts, grads = [], []
    try:
        for i in range(0, len(paths), args.batch_size):
            chunk = paths[i:i + args.batch_size]
            batch = torch.stack([tf(model_view(p, args.image_sz)) for p in chunk]).to(device)
            g = geo[i:i + args.batch_size].to(device) if geo is not None else None
            with torch.enable_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == "cuda"):
                out = model(batch, g) if g is not None else model(batch)
                logits = out[0] if isinstance(out, tuple) else out
                act = store["act"]
                acts.append(act.detach()[:, prefix:, :].mean(dim=1).float().cpu().numpy())
                if want_grad_for is not None:
                    idx = torch.as_tensor(want_grad_for[i:i + args.batch_size], device=device)
                    model.zero_grad(set_to_none=True)
                    logits.gather(1, idx[:, None]).sum().backward()
                    grads.append(act.grad[:, prefix:, :].mean(dim=1).float().cpu().numpy())
    finally:
        handle.remove()
        backbone.set_grad_checkpointing(was_ckpt)
        model.zero_grad(set_to_none=True)
    A = np.concatenate(acts)
    G = np.concatenate(grads) if grads else None
    return A, G


def train_cav(acts: np.ndarray, labels: np.ndarray, seed: int, C: float = 0.01):
    """Concept direction: the normal to a regularised linear boundary.

    Regularisation is not optional here. With 1024 activation dims and a few
    hundred specimens the classes are linearly separable whatever the labels
    mean, so an unregularised fit scores 1.0 in-sample on concept and shuffled
    labels alike, and its normal points in an arbitrary direction.
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000, random_state=seed, C=C).fit(acts, labels)
    v = clf.coef_[0]
    return v / (np.linalg.norm(v) + 1e-12), clf


def stage_cav(args) -> None:
    out = Path(args.out)
    meta = pd.read_csv(out / META_NAME)
    lab = pd.read_csv(out / args.labels if not Path(args.labels).is_absolute()
                      else Path(args.labels))
    df = meta.merge(lab, on="fname")
    device = torch.device(args.device)
    model, nameslist, temperature, geo_dim, label_level = build_full_model(
        args.checkpoint, device)
    backbone = getattr(model, "backbone", model)
    _, layer_idx = layer_module(backbone, args.layer)
    names = {n: i for i, n in enumerate(nameslist)}
    print(f"  CAVs at block {layer_idx} of {len(backbone.blocks)}")

    rng = np.random.RandomState(args.seed)
    results = []
    for concept, grp in df.groupby("concept"):
        if args.concept and concept != args.concept:
            continue
        grp = grp.sample(min(len(grp), args.max_specimens), random_state=args.seed)
        grp = grp.reset_index(drop=True)
        y = grp["value"].values.astype(int)
        cls = grp["species"].map(lambda s: names.get(s, -1)).values
        keep = cls >= 0
        grp, y, cls = grp[keep].reset_index(drop=True), y[keep], cls[keep]
        # Activations are the expensive part (a forward AND backward pass per
        # specimen); the null is just logistic regressions on top of them. Cache
        # them so a bigger null costs seconds instead of another GPU pass.
        cache = out / f"cav_cache_{concept}_L{layer_idx}.npz"
        # Identity, not row count: with --max-specimens capping most concepts at
        # the same number, a changed label file produces a cache of exactly the
        # same shape holding entirely different specimens.
        fingerprint = np.array(grp["fname"].tolist())
        if args.reuse_cache and cache.exists():
            z = np.load(cache, allow_pickle=False)
            if ("fnames" in z and len(z["fnames"]) == len(fingerprint)
                    and (z["fnames"] == fingerprint).all()):
                A, G = z["A"], z["G"]
                print(f"    {concept}: activations from cache")
            else:
                print(f"    {concept}: cache is for different specimens — recomputing")
                z = None
        else:
            z = None
        if z is None:
            geo = (encode_coords(grp["decimalLatitude"], grp["decimalLongitude"])
                   if geo_dim else None)
            A, G = activations(model, backbone, grp["path"].tolist(), geo, args.layer,
                               args, device, want_grad_for=cls)
            np.savez_compressed(cache, A=A, G=G, fnames=fingerprint)

        v, clf = train_cav(A, y, args.seed, args.cav_C)
        # Held-out, not in-sample: see train_cav on why in-sample is always 1.0.
        split = rng.permutation(len(y))
        cut = int(0.7 * len(split))
        _, clf_ho = train_cav(A[split[:cut]], y[split[:cut]], args.seed, args.cav_C)
        holdout = float(clf_ho.score(A[split[cut:]], y[split[cut:]]))
        # Cross-genus control. Species-level labels let a CAV cheat by encoding
        # the taxa themselves; if it still separates the concept on genera it
        # never saw, it is tracking something more general than taxon identity.
        genera = grp["genus"].values
        uniq = np.unique(genera)
        # Retry the draw: one unlucky split (all-positive test genera, too few
        # specimens) is not a reason to report nothing for the control that
        # matters most. Averaged over the splits that are usable.
        scores = []
        for _ in range(args.cross_genus_tries):
            held = rng.permutation(uniq)[:max(1, len(uniq) // 3)]
            te = np.isin(genera, held)
            if te.sum() < 10 or (~te).sum() < 20:
                continue
            if len(np.unique(y[~te])) < 2 or len(np.unique(y[te])) < 2:
                continue
            _, clf_tr = train_cav(A[~te], y[~te], args.seed, args.cav_C)
            scores.append(float(clf_tr.score(A[te], y[te])))
        cross = float(np.mean(scores)) if scores else np.nan

        sens = G @ v
        tcav = float((sens > 0).mean())
        # Null: the same pipeline with the concept labels shuffled. Anything the
        # geometry of this activation space produces on its own shows up here.
        # The null gets its own generator, seeded per concept. Sharing `rng`
        # with the rest of the loop made the z-score move between runs whenever
        # anything upstream consumed a different number of draws — the estimate
        # shifted while the data did not.
        null_rng = np.random.default_rng(args.null_seed + abs(hash(concept)) % 10_000)
        null = []
        for _ in range(args.null_runs):
            vr, _ = train_cav(A, null_rng.permutation(y), args.seed, args.cav_C)
            null.append(float(((G @ vr) > 0).mean()))
        null = np.array(null)
        z = (tcav - null.mean()) / (null.std() + 1e-9)
        results.append({
            "concept": concept, "n": len(grp), "n_species": grp.species.nunique(),
            "n_genera": len(np.unique(genera)),
            "cav_holdout_accuracy": round(holdout, 3),
            "cross_genus_accuracy": round(cross, 3) if np.isfinite(cross) else np.nan,
            "tcav": round(tcav, 3), "null_mean": round(float(null.mean()), 3),
            "null_sd": round(float(null.std()), 3), "z": round(float(z), 2),
        })
        print("   ", results[-1], flush=True)

    res = pd.DataFrame(results)
    res.to_csv(out / "tcav_results.csv", index=False)
    print("\n" + res.to_string(index=False))
    print("\n  Read cross_genus_accuracy (does the direction survive on genera the "
          "CAV never saw?) and z (is TCAV distinguishable from shuffled labels?). "
          "A concept can be clearly decodable and still have no demonstrable "
          "effect on the logits — those are different claims.")
    print(f"  Wrote {out / 'tcav_results.csv'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp, ckpt=False):
        sp.add_argument("--out", required=True)
        sp.add_argument("--seed", type=int, default=42)
        if ckpt:
            sp.add_argument("--checkpoint", required=True)
            sp.add_argument("--image-sz", type=int, default=640)
            sp.add_argument("--batch-size", type=int, default=8)
            sp.add_argument("--device",
                            default="cuda" if torch.cuda.is_available() else "cpu")

    t = sub.add_parser("from-treatments", help="Traits from TaxPub treatment XML.")
    common(t)
    t.add_argument("--xml", nargs="*", default=[], help="TaxPub treatment XML.")
    t.add_argument("--pdf", nargs="*", default=[],
                   help="Plain PDF revisions with a text layer.")
    t.add_argument("--min-specimens", type=int, default=60)
    t.set_defaults(func=stage_from_treatments)

    a = sub.add_parser("auto-concepts", help="Proxy concepts from the PC1 masks.")
    common(a)
    a.set_defaults(func=stage_auto_concepts)

    m = sub.add_parser("template", help="Contact sheets + CSV for manual scoring.")
    common(m)
    m.add_argument("--n", type=int, default=120)
    m.add_argument("--per-page", type=int, default=12)
    m.add_argument("--dpi", type=int, default=140)
    m.add_argument("--traits", default=",".join(TRAITS))
    m.set_defaults(func=stage_template)

    c = sub.add_parser("cav", help="Train CAVs and score TCAV.")
    common(c, ckpt=True)
    c.add_argument("--labels", required=True, help="CSV of fname,concept,value.")
    c.add_argument("--layer", type=int, default=-6)
    c.add_argument("--concept", help="Only this concept.")
    c.add_argument("--max-specimens", type=int, default=400)
    c.add_argument("--null-runs", type=int, default=200,
                   help="Shuffled-label CAVs for the null. 20 is far too few — "
                        "the z-score then moves by a whole point between runs.")
    c.add_argument("--null-seed", type=int, default=1234)
    c.add_argument("--reuse-cache", action="store_true",
                   help="Reuse cached activations for this concept and layer.")
    c.add_argument("--cross-genus-tries", type=int, default=8,
                   help="Random held-out-genus draws to average the control over.")
    c.add_argument("--cav-C", type=float, default=0.01,
                   help="Inverse regularisation for the CAV fit. Small on purpose.")
    c.set_defaults(func=stage_cav)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
