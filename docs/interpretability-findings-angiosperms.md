# What the family classifier keys on

Findings from re-running the interpretability study on a deliberately different
dataset and model, 28 August 2026. The companion study on the Annonaceae species
classifier is `interpretability-findings.md`; this document reports the same
probes on an all-angiosperm family classifier and says where the two agree.

**Model.** `acc-epoch=07-val_Accuracy=0.9498.ckpt` (trained 20 July 2026):
`vit_large_patch16_dinov3.lvd1689m` at 640 px, a **single head over 235
families**, non-hierarchical, softmax temperature 0.531. Held in R2 under
`Angiosperm-families_Africa/checkpoints/`. Note this is *not* the checkpoint
sitting in the local `cloud_results/` directory (`acc-epoch=06`, 9 May): that
earlier one predates commit 71dec5d and carries no `split` payload, so its
held-out set cannot be recovered.

**What makes it a good contrast.** Same backbone, same image size, same 40×40
token grid — so patch-level results are directly comparable — but a different
rank (family, not species), 7× the held-out set, and **no geographic fusion at
all** (`use_location=False`, no `geo_mlp` in the state dict). Any sheet-derived
signal found here cannot be coordinates leaking in through the geo channel.

**Evaluation set.** The 15,092 held-out specimens in the checkpoint's own
`split` payload (60,365 train / 15,092 validation, seed 42), covering 235
families and 6,094 species. The set is near-balanced — capped at 100 specimens
per family, median 90, minimum 4 — so **the majority-class baseline is 0.0066**
and uniform chance is 0.0043. Every accuracy below should be read against
0.0066, not against the 10–20% intuition a long-tailed set would give.

**Reproduce.** Branch `interpretability-probe`; scripts `probe_embeddings.py`,
`probe_confounds.py`, `probe_perturbations.py`, `probe_saliency.py`. All import
preprocessing from `identify_herbarium.py`, so each probe sees exactly what
inference sees. Outputs in `<r2_restore>/interp/`.

---

## Summary

Every headline conclusion from the Annonaceae study reproduces at family rank,
several of them to two decimal places. The model reads the plant, at fine
spatial scale, largely without colour, largely ignoring the sheet's global
layout. Measured baseline on the held-out set is **94.9% top-1, 98.4% top-5**.

The new result is about the label. At 640 px the determination text *is*
legible, some sheets print the family outright, and the model demonstrably
reads it — and it still makes almost no difference to what the model predicts.

| Question | Answer |
|---|---|
| Does it need the plant? | Yes — blanking it costs 63.9 points |
| Does it need colour? | Barely — grayscale costs 2.2 |
| Does it need fine detail? | Critically — blur σ=4 costs 31.3 |
| Does it need global architecture? | No — 128 px patch shuffle costs 3.1 |
| Can it work from outlines alone? | Substantially — Sobel edges retain 50.9% |
| Does it know the herbarium? | Yes — 92.9% from a species-disjoint probe |
| **Can it read the label?** | **Yes — label text alone gives 18× baseline** |
| **Does it use the label?** | **Barely — blurring the type costs 0.5** |

---

## 1. Perturbation battery

Baseline 0.9491. Each condition destroys one class of information; the model's
own preprocessing is applied afterwards, so nothing here is out of frame.

| condition | top-1 | Δ | condition | top-1 | Δ |
|---|---|---|---|---|---|
| baseline | 0.9491 | — | resolution 320 | 0.9285 | −2.1 |
| grayscale | 0.9274 | −2.2 | resolution 224 | 0.8849 | −6.4 |
| saturation ×0.5 | 0.9461 | −0.3 | resolution 160 | 0.7983 | −15.1 |
| saturation ×1.5 | 0.9472 | −0.2 | resolution 112 | 0.6159 | −33.3 |
| hue 60° | 0.9066 | −4.3 | resolution 64 | 0.2568 | −69.2 |
| hue 120° | 0.9033 | −4.6 | shuffle 128 px | 0.9182 | −3.1 |
| blur σ1 | 0.9404 | −0.9 | shuffle 64 px | 0.8700 | −7.9 |
| blur σ2 | 0.8949 | −5.4 | shuffle 32 px | 0.7370 | −21.2 |
| blur σ4 | 0.6358 | −31.3 | shuffle 16 px | 0.4097 | −53.9 |
| blur σ8 | 0.2293 | −72.0 | phase scramble | 0.4520 | −49.7 |
| blur σ16 | 0.0495 | −90.0 | **Sobel edges** | **0.5085** | −44.1 |

**Colour is nearly free.** Grayscale costs 2.2 points and saturation nothing
measurable. Annonaceae: 2.5 and ~0.

**Fine-to-mid-scale structure is everything**, and the blur and resolution
ladders are the same ladder — σ4 gives 0.6358, 112 px gives 0.6159. The
Annonaceae run found the identical correspondence (both landed on 0.4842).

**Global layout barely matters.** Shuffling 128 px blocks costs 3.1 points —
matching Annonaceae's 3.1 exactly. Whole-sheet architecture, including internode
positions across the sheet, is not the cue at either taxonomic rank. Degradation
only becomes severe once the shuffle block drops near the 16 px patch size and
starts destroying local structure itself.

**Edges alone retain 0.5085**, better than Annonaceae's 0.427. This is the
quantitative form of the observation that the model can read line drawings:
strip every surface and keep only outlines and it still calls half of 235
families. It also cannot be a label effect — the Sobel transform destroys text
along with everything else.

One condition deserves follow-up: **phase scramble retains 0.4520**, more than a
texture-only control ought to.

---

## 2. Is it reading the plant, or the sheet?

Plant masks come from PC1 of the patch tokens (§3), which is a foreground
detector: it stays at sheet brightness over labels, barcodes and colour charts.
The plant mask covers 45.3% of the sheet on average (p10 34.1%, p90 57.2%).

| condition | top-1 | Δ vs baseline | Δ vs shape control |
|---|---|---|---|
| baseline | 0.9491 | — | |
| plant_removed | 0.3105 | −63.9 | −60.7 |
| plant_only | 0.9092 | −4.0 | −0.8 |
| shift_control | 0.9176 | −3.2 | 0.0 |
| quadrant_removed | 0.9339 | −1.5 | +1.6 |
| quadrant_only | 0.5845 | −36.5 | −33.3 |

The plant does the work: removing it costs 63.9 points, while keeping *only* the
plant costs 4.0 — and against an area- and shape-matched mask placed elsewhere
(3.2), the marginal cost of losing the entire sheet is 0.8 points. Removing the
label quadrant costs 1.5.

**`quadrant_only` = 0.5845 is not a label measurement.** The bottom-right
quadrant contains plant material on most sheets. Isolating the label properly
(§2b) drops it to 0.1181 — the naive reading overstates the label by 5×.

### The institution probe

A logistic probe on the pooled embedding, split by species so no test species is
seen in training (10,460 train / 4,632 test):

| probe | test accuracy |
|---|---|
| **embedding → institution** | **0.9285** |
| **embedding → family** (same species-disjoint split) | **0.8029** |
| embedding → species (reference, random split) | 0.3564 |
| genus one-hot → institution | 0.2971 |
| majority class (institution) | 0.2610 |
| embedding → shuffled institution | 0.2170 |

**On the same rows and the same split, the embedding identifies the herbarium
better than it identifies the plant's family** — 0.9285 against 0.8029. The
genus one-hot baseline at 0.2971 rules out this being taxon/institution
confounding. Annonaceae showed the same inversion, milder (0.894 vs 0.644).

This model has no geo channel, so none of the signal is coordinates. It is the
sheet itself: paper, mounting style, stamps, tape, colour chart, label design.

The geo ablation stage does not apply here and exits cleanly — there is no
`geo_mlp` to ablate.

---

## 2b. Does it read the label text?

Motivated by an unexpected observation: at 640 px the determination text is
**legible**. `Resize` scales the short side, so a 750×1200 sheet lands near
0.85 px/mm and a 16 px patch covers a word or two. Some sheets print the answer
outright — one *Cassytha filiformis* sheet carries "LAURAC." in plain type.

Blanking cannot separate *reading the determination* from *recognising the
herbarium's paper and layout*, because both live in the same quadrant. The
`text` stage separates them by blurring one region and blanking another:
`label_only` is the quadrant **minus the plant**, so no leaf is in frame, and
`label_only_blur` applies σ=3 — enough to destroy 8–10 px type while leaving
paper tone, rules, stamps and layout intact.

| condition | top-1 | vs majority (0.0066) |
|---|---|---|
| baseline | 0.9491 | |
| **text_blur_insitu** | **0.9441** | type destroyed on an intact sheet: **−0.50** |
| plant_only (§2) | 0.9092 | |
| plant_only_blur | 0.6653 | blur control: −24.4 |
| **label_only** | **0.1181** | 18× |
| **label_only_blur** | **0.0427** | 6.5× |

**It reads the label.** The label region with every plant pixel removed calls
family at 0.1181, eighteen times the majority baseline, and destroying the type
takes it to 0.0427. About two-thirds of the label region's family signal is the
writing itself.

**The blur control rules out the lazy explanation.** Blur is out of distribution,
so a drop under blur means nothing until you know what the same blur costs a
channel carrying no text. It costs the plant channel 27% of its accuracy
(0.9092 → 0.6653) and the label channel 64% of its own (0.1181 → 0.0427). The
label's collapse is disproportionately about text, not about blur.

**It does not use the label.** Blurring the type on an otherwise untouched sheet
costs half a point, 0.9491 → 0.9441, and part of even that is the blur artefact
rather than the text. So ≤0.5% of specimens can be text-dependent. The plant
channel alone holds 0.9092; the label is a redundant copy that never has to be
consulted.

**Redundancy is why single ablations were not enough here.** Two channels that
are each sufficient both look cheap to remove. `quadrant_removed` costing 1.5
points was consistent both with "the label is unused" and with "the label is
used but duplicated" — only isolating the channel and degrading it *within*
itself separates them.

The residual is not mainly the label. `plant_removed` (whole sheet, plant gone,
every label present) scores 0.3105, while the label quadrant minus plant gives
0.1181. **Most of the non-plant signal lies outside the label quadrant** — the
same unexplained residual the Annonaceae study left open at 0.095, and
consistent with the institution probe at 0.9285.

---

## 3. Representation structure

**Nearest-neighbour retrieval** on the pooled embedding (cosine, held-out only):

| field | top-1 agreement | chance | lift |
|---|---|---|---|
| species | 0.4219 | 0.0004 | 1006× |
| genus | 0.7127 | 0.0021 | 334× |
| **family** | **0.8510** | 0.0058 | 146× |
| institutionCode | 0.7116 | 0.1875 | 3.8× |
| countryCode | 0.5405 | 0.1690 | 3.2× |

Retrieval alone reaches 0.8510 at family against a trained head's 0.9491. The
institution lift (3.8×) is markedly higher than Annonaceae's (2.8×), and here it
cannot be geo-mediated.

**PC1 of the patch tokens** is again a plant-vs-sheet foreground detector,
reproducing the Annonaceae finding on a completely different taxonomic sample.
It stays at sheet brightness over mounted determination slips, barcodes and
colour charts — which is precisely what makes the §2b label isolation possible.

**UMAP** of the pooled embedding is in `projection_image.png`, panelled by
family, genus, institution and country.

---

## 4. Where the evidence sits

`saliency_edges.png`: eight families, half of them robust to the `edges`
condition and half fragile, each shown as the model's view, an occlusion
sensitivity map [1] and an attention rollout [2].

**Occlusion evidence sits on the plant.** Across all eight, the cells whose
occlusion costs p(true) fall on stems, leaf clusters and flowering shoots —
Vahliaceae on the inflorescence, Malpighiaceae on the central stem junction,
Grubbiaceae on the shoot. None of it sits on labels or barcodes, which stay
neutral. This is the direct test of *use*, and it agrees with §2 and §2b:
the classifier's evidence is the plant.

**Attention rollout says something different, and that difference is the
point.** Rollout reliably lights up the colour chart strip and the label text
rows — most extremely on the Frankeniaceae sheet, whose top-left colour chart
dominates the map. The backbone *looks at* the sheet furniture; the classifier
does not *use* it. Rollout answers "what did attention flow through", occlusion
answers "what would change the answer", and only the second is evidence about
the decision. Both the Annonaceae run and this one make the same point on
completely different taxa.

Rollout is again dominated by a few high-norm outlier tokens — the top 5 hold
11–16% of the mass, the same 11–16% seen on Annonaceae — so it needs percentile
clipping to be legible at all.

**Negative occlusion cells are prominent here.** Many cells are blue: covering
them *raises* p(true), meaning part of the specimen actively misleads the model.
Cardiopteridaceae shows this across most of its leaf area despite p(true)=0.88.
The Annonaceae study flagged this as worth following up; at 15,092 specimens and
235 families it is clearly not a one-off, and it is the most interesting open
thread from either study.

---

## Methodological notes

Additions to the notes in the companion document.

- **A quadrant is not a label.** `quadrant_only` = 0.5845 read as a label
  measurement overstates the label 5× — the quadrant is full of plant. Region
  masks must be intersected with the foreground mask before they measure what
  their name claims.
- **Cheap-to-ablate does not mean unused.** When two channels are each
  sufficient, removing either one costs almost nothing. Redundancy has to be
  broken from inside the channel, not by deletion.
- **Check what the model can actually see before arguing from resolution.** The
  arithmetic said 640 px was far too coarse for OCR. Rendering the model's own
  view showed the type plainly readable. One `model_view()` call settled it, and
  the prior was wrong.
- **Read accuracies against the actual class prior.** This held-out set is
  capped at 100/family, so the majority baseline is 0.0066, not the ~10% a
  long-tailed set would give. `label_only` = 0.1181 is 18× baseline, not the
  "barely above chance" it would be under the wrong prior.
- **`--n` is unusable through `conda run`**, which parses it as `--name` /
  `--no-plugins`. Invoke the env's python directly for probes taking short flags.
- **`conda run` buffers all output until exit**, so a long stage looks silent and
  dead. Watch the GPU, not the log.

---

## Still open

- **Negative occlusion cells** — covering part of a specimen raises p(true).
  Prominent at family rank across many sheets; no explanation yet.
- **The non-plant residual.** `plant_removed` = 0.3105 with only 0.1181
  attributable to the label quadrant. What carries the rest is unidentified —
  mounting style, paper, stamps and tape are the candidates, and the institution
  probe at 0.9285 says the information is certainly there.
- **Phase scramble at 0.4520**, higher than a texture-only control should give.
- **Concept probes (step 5) do not port.** The Annonaceae CAVs came from
  PhytoKeys/Blumea trait treatments; there is no equivalent trait vocabulary
  spanning 235 families, so this study covers steps 1–4 only.
- **Silhouette** was skipped in §1: it needs `plant_masks.npz` from §2, which was
  still running. The companion study found it untrustworthy anyway (out of
  distribution), with `edges` the reliable shape-only condition.

## References

[1] Zeiler, M.D. & Fergus, R. (2014). Visualizing and understanding
convolutional networks. *ECCV 2014*, 818–833. (Occlusion sensitivity.)

[2] Abnar, S. & Zuidema, W. (2020). Quantifying attention flow in transformers.
*ACL 2020*, 4190–4197. (Attention rollout.)
