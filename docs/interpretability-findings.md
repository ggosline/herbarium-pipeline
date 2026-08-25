# What the herbarium classifier keys on

Findings from a five-part interpretability study of the Annonaceae species
classifier, run 24 August 2026.

**Model.** `acc-epoch=10-val_Accuracy=0.8143.ckpt` (trained 25 July 2026):
`vit_large_patch16_dinov3.lvd1689m` at 640 px, hierarchical heads over 261
species and 40 genera, 64-dimensional geographic feature fusion, softmax
temperature 0.546 [7]. Checkpoint held in R2 under `Annonaceae/checkpoints/`.

**Evaluation set.** The 2,177 held-out specimens recorded in the checkpoint's
own `split` payload (8,705 train / 2,177 validation, seed 42). Nothing below is
measured on training images — the model memorises those (100% accurate, mean
confidence 0.9985), so any figure averaging them in reads far too well.

**Reproduce.** Branch `interpretability-probe`; scripts `probe_embeddings.py`,
`probe_confounds.py`, `probe_perturbations.py`, `probe_saliency.py`,
`probe_concepts.py`. All import preprocessing from `identify_herbarium.py`, so
each probe sees exactly what inference sees.

---

## Summary

The classifier reads the plant, at fine spatial scale, largely without colour,
and largely without regard to the sheet's global layout. It carries strong
information about which herbarium mounted a sheet, and demonstrably does not
use it. Baseline accuracy on the held-out set is **81.1% top-1, 94.3% top-5**.

| Question | Answer |
|---|---|
| Does it need the plant? | Yes — blanking it costs 71.6 points |
| Does it need the sheet, labels, barcodes? | No — blanking them costs 2.8 |
| Does it need colour? | Barely — grayscale costs 2.5 |
| Does it need fine detail? | Critically — blur σ=4 costs 32.7 |
| Does it need global architecture? | No — 128 px patch shuffle costs 3.1 |
| Can it work from outlines alone? | Partly — Sobel edges retain 42.7% |
| Does it know the herbarium? | Yes — 89.4% from a species-disjoint probe |
| Does it use the herbarium? | No — removing the label quadrant costs 1.2 |

---

## 1. Perturbation battery

Each condition destroys one class of information inside the model's own framing
(applied after resize and centre crop, before normalisation) and is scored on
the species head. Geo is held at real values throughout, so a condition that
destroys the image floors at the geo-only prior, not at chance.

| condition | top-1 | Δ | condition | top-1 | Δ |
|---|---|---|---|---|---|
| baseline | 0.811 | — | resolution 320 | 0.785 | −0.026 |
| grayscale | 0.786 | −0.025 | resolution 224 | 0.736 | −0.074 |
| saturation ×0.5 | 0.809 | −0.001 | resolution 160 | 0.643 | −0.168 |
| saturation ×1.5 | 0.806 | −0.005 | resolution 112 | 0.484 | −0.327 |
| hue +60° | 0.778 | −0.033 | resolution 64 | 0.220 | −0.591 |
| hue +120° | 0.750 | −0.061 | phase scramble | 0.238 | −0.573 |
| blur σ=1 | 0.792 | −0.019 | shuffle 128 px | 0.780 | −0.031 |
| blur σ=2 | 0.721 | −0.090 | shuffle 64 px | 0.695 | −0.116 |
| blur σ=4 | 0.484 | −0.327 | shuffle 32 px | 0.499 | −0.312 |
| blur σ=8 | 0.168 | −0.643 | shuffle 16 px | 0.164 | −0.647 |
| blur σ=16 | 0.046 | −0.765 | Sobel edges | 0.427 | −0.384 |

**Colour is nearly free.** Grayscale costs 2.5 points and saturation scaling is
within noise. Only a 120° hue rotation reaches 6 points. This matters
practically: specimen colour shifts with drying age, mounting era and scanner,
and the model is largely invariant to it.

**Fine detail is decisive.** Accuracy falls off a cliff past blur σ=2, and the
resolution ladder mirrors it — 112 px lands on 0.4842, the same value as blur
σ=4, two independent routes to destroying the same band of information. 224 px
is survivable at a cost of 7 points; below 160 px the signal is being thrown
away.

**Global architecture is not what it keys on.** Scrambling the sheet into
128 px blocks destroys every global relationship — habit, internode spacing,
the arrangement of the branch on the sheet — and costs **3.1 points**. The
collapse at 16 px (the ViT patch size) is destruction of local structure
itself, not of layout. This is the direct answer to whether the model reads
whole-plant architecture: it does not.

**It can work from line drawings.** Sobel edges alone, with all colour and tone
gone, retain 42.7%.

**One number not to trust.** A silhouette-only condition (PC1 mask, black on
white) scores 0.010, but that is out-of-distribution collapse — no texture, no
sheet, nothing the model has ever seen — not a measurement of how informative
an outline is. Sobel edges is the trustworthy shape-only condition. The same
trap appears in §3.

**Per species** (168 species with n≥8): grayscale robustness tracks baseline
accuracy tightly (Spearman +0.87), but edge and blur robustness only loosely
(+0.42, +0.46). Some taxa survive heavy degradation far better than their
accuracy predicts — *Annona glabra*, *Monanthotaxis buchananii* and
*Sphaerocoryne gracilis* hold 100% on edges alone; *Monanthotaxis vogelii*
falls from 100% to 0%.

---

## 2. Is it reading the plant, or the sheet?

Occlusion of the specimen versus the sheet furniture, scored on the species
head. The plant mask is thresholded from PC1 of the patch tokens (§3); its
orientation is pinned by geometry, not by PCA sign, since the sheet dominates a
scan's border.

| condition | top-1 | vs baseline | vs area-matched control |
|---|---|---|---|
| baseline | 0.811 | — | +0.044 |
| **plant removed** | **0.095** | **−0.716** | **−0.672** |
| plant only | 0.783 | −0.028 | +0.016 |
| shift control (same mask, wrong place) | 0.767 | −0.044 | 0 |
| label quadrant removed | 0.798 | −0.012 | +0.032 |
| label quadrant only | 0.448 | −0.362 | −0.318 |

The control is what makes this readable: an equal-area, equal-shape hole placed
at random costs 4.4 points by itself, so every condition is reported against it
as well as against baseline. Blanking the plant destroys the model (90% of
predictions change). Blanking everything *but* the plant costs 2.8 points and
is 1.6 points *better* than a random hole of the same size.

`quadrant_only` is not a labels-only condition: measured against the masks,
13.2% of a typical sheet's plant falls in the bottom-right quadrant and that
quadrant is 28.9% plant, so its 44.8% is mostly specimen.

**But the embedding does know the herbarium.** A logistic probe on the pooled
embedding, trained and tested on **disjoint species** so it cannot win by
memorising taxa:

| probe | test accuracy |
|---|---|
| embedding → institutionCode | **0.894** |
| genus one-hot → institutionCode (confound baseline) | 0.317 |
| majority class | 0.330 |
| embedding → shuffled institution (sanity) | 0.204 |
| embedding → species (reference, random split) | 0.644 |

The embedding identifies the *herbarium* better than it identifies the
*species*, 58 points above what genus alone buys. Note this corrected an
earlier reading: the UMAP of §3 shows no global institution structure, which
looked like evidence that the retrieval-level institution signal was mere
taxon confounding. The probe finds it in directions a two-dimensional
projection discards. Presence in the representation and use by the classifier
are different claims, and only the second is ruled out here.

**Geographic features.** Coordinates are worth ~4.8 points to the specimens
that have them (43.5% of the held-out set):

| condition | overall | with coords | no coords |
|---|---|---|---|
| real | 0.810 | 0.799 | 0.818 |
| no location | 0.789 | 0.751 | 0.818 |
| permuted | 0.758 | 0.724 | 0.784 |
| permuted within country | 0.801 | 0.775 | 0.821 |
| all zeros (OOD control) | 0.537 | 0.447 | 0.606 |

Wrong coordinates are worse than none (0.724 vs 0.751) — geo actively misleads
rather than being ignored. Nearly all the benefit is coarse: permuting within a
country costs only 2.4 points, so the model uses "which country" far more than
"which spot". The no-coordinate column is a built-in check — `no_location`
leaves it unchanged at 0.818, as it must.

The all-zeros row is **not** a valid ablation and is labelled OOD in the output.
`encode_coords` maps a missing coordinate to `[1, 0, 0, 0]`, not to zeros: it
sets lat/lon to 0 and takes cos(0)·cos(0) = 1, with the fourth element flagging
"no location". Feeding all-zeros is an input the model never saw in training,
and its 0.537 measures collapse off-distribution.

---

## 3. Representation structure

**Patch-token PCA.** PCA of *activations*, not weights, over the 40×40 token
grid. PC1 is a foreground detector: bright on leaf tissue, dark on the mounting
sheet, and dark on labels, barcodes and colour charts. PC1–3 explain 8.0 / 4.3 /
3.3% of token variance, so this is the coarsest structure, not the whole story.

**UMAP** [5] of the pooled embedding: genus forms clean contiguous territories
(*Xylopia*, *Uvaria*, *Monanthotaxis*, *Artabotrys*, *Isolona*, *Uvariopsis*,
*Piptostigma*), while `institutionCode` is visually structureless on the same
layout — see §2 for why that is not the whole picture.

**Nearest-neighbour retrieval** (cosine, top-1 agreement against chance):

| field | image only | + geo | chance |
|---|---|---|---|
| species | 0.610 | 0.604 | 0.004 |
| genus | 0.876 | 0.851 | 0.078 |
| institutionCode | 0.558 | 0.582 | 0.201 |
| countryCode | 0.523 | 0.560 | 0.113 |

Appending the geo vector makes retrieval *worse* taxonomically (genus −2.5
points) and better geographically, and detaches a 189-specimen West African
island — 100% coordinate-bearing, spanning many genera — showing geo can
override taxonomy for a densely sampled region.

---

## 4. Where the evidence sits

Occlusion sensitivity [1] and attention rollout [2], side by side.

Occlusion evidence sits on the plant — leaf blades, shoots, flower and fruit
clusters — and not on labels or barcodes. Attention rollout on the same
specimens reliably lights up the **colour chart strip and label text rows**. The
backbone *looks* at sheet furniture; the classifier does not *use* it,
consistent with the 1.2-point cost of removing the label quadrant. Rollout
answers "where does information flow", not "what changes the answer", and
should not be read as an explanation on its own.

Two mechanics worth recording. timm implements DINOv3 as `EvaAttention` with
fused SDPA, which never materialises the attention matrix: `fused_attn` must be
disabled per block before hooking, or the hook silently captures nothing. And
the rollout map is dominated by a few very high-norm outlier tokens — the
register-token artefact described for DINOv2 [4] — with the top 5 tokens
holding 11–16% of the mass, so it needs percentile clipping to be legible.

---

## 5. Botanical concepts (TCAV)

Concept activation vectors [3] at block 18 of 24. The last layer is unusable
for this: the head is linear, so the directional derivative there is identical
for every specimen and the score collapses to 0 or 1.

Concepts are taken from published taxonomic treatments [8][9][10] rather than
scored by hand — 137 of 261 species, covering 1,199 of 2,177 specimens.

| concept | held-out acc | cross-genus acc | TCAV | null | z |
|---|---|---|---|---|---|
| many_veins | 0.725 | 0.722 | 0.698 | 0.503 ± 0.100 | 1.95 |
| large_blade | 0.933 | 0.825 | 0.657 | 0.498 ± 0.098 | 1.63 |
| blade_hairy | 0.742 | 0.667 | 0.537 | 0.498 ± 0.099 | 0.40 |
| elongate_blade | 0.775 | 0.688 | 0.530 | 0.497 ± 0.100 | 0.33 |
| habit_liana | 0.883 | 0.729 | 0.517 | 0.495 ± 0.105 | 0.22 |

**The traits are encoded.** Cross-genus accuracy — train the direction on some
genera, test on genera the CAV has never seen — runs 0.67–0.83 against a 0.5
baseline. Blade size (0.83) and secondary vein density (0.72) are real
directions that generalise beyond the taxa used to define them. Vein density
being present is coherent with §1: it is exactly the fine-scale character that
survives 224 px but not 112 px.

**That they drive predictions is not established.** No z-score reaches 2 across
five concepts, and the evidence weakened as data was added: on a smaller
concept set (110 species) `blade_hairy` scored z=1.78 and `elongate_blade`
z=0.90; adding the *Monanthotaxis* revision dropped them to 0.40 and 0.33 while
cross-genus decodability held steady. An effect that dissolves when the sample
grows was not an effect.

Two caveats, both of which bias *towards* the null rather than away: this is a
global TCAV variant (per specimen, does the concept direction raise that
specimen's own class logit) where standard TCAV scores one target class at a
time, and species-level labels carry irreducible noise because a treatment
describes the taxon while a given sheet may not show the character at all.

**`habit_liana` is the most informative row**: 0.73 cross-genus decodability
with a TCAV score at the null. The model can tell a liana from a tree and does
not use it — the same representation-versus-use split found in §2 and §4,
reached a third independent way. It also shows the method returns zero when
zero is the answer, which is what makes the non-zero rows worth anything.

---

## Methodological notes

Each of these produced a wrong number before it produced a right one.

- **Controls before conclusions.** "Accuracy fell when I removed the plant" is
  meaningless without an area- and shape-matched mask elsewhere; that control
  alone costs 4.4 points.
- **Out-of-distribution inputs masquerade as ablations.** Both the all-zeros geo
  vector (0.537) and the silhouette-only image (0.010) look like devastating
  ablations and are artefacts of feeding the model something it never saw.
- **Regularise concept probes.** With 1,024 dimensions and a few hundred
  specimens the classes are linearly separable whatever the labels mean; an
  unregularised fit scores 1.000 in-sample on real and shuffled labels alike.
- **Give the null its own generator, and enough draws.** At 20 shuffles, and
  sharing the pipeline's RNG, one z-score moved a full point between runs while
  the data stood still.
- **A projection is not the representation.** UMAP showing no institution
  structure did not mean institution was absent; a linear probe found it at
  0.894.
- **Check fill rates, not exceptions.** Trait patterns fitted to one flora
  extracted *nothing* from another while other fields kept parsing, so the
  output looked healthy.

---

## References

[1] Zeiler, M.D. & Fergus, R. (2014). Visualizing and understanding
convolutional networks. *ECCV 2014*, 818–833. (Occlusion sensitivity.)

[2] Abnar, S. & Zuidema, W. (2020). Quantifying attention flow in transformers.
*ACL 2020*, 4190–4197. (Attention rollout.)

[3] Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viégas, F. &
Sayres, R. (2018). Interpretability beyond feature attribution: quantitative
testing with concept activation vectors (TCAV). *ICML 2018*, 2668–2677.

[4] Darcet, T., Oquab, M., Mairal, J. & Bojanowski, P. (2024). Vision
transformers need registers. *ICLR 2024*. (High-norm outlier tokens.)

[5] McInnes, L., Healy, J. & Melville, J. (2018). UMAP: Uniform Manifold
Approximation and Projection for dimension reduction. arXiv:1802.03426.

[6] Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning
robust visual features without supervision. *TMLR*. (Backbone lineage; the
checkpoint uses the DINOv3 LVD-1689M weights as distributed by `timm`.)

[7] Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017). On calibration of
modern neural networks. *ICML 2017*, 1321–1330. (Temperature scaling.)

[8] Couvreur, T.L.P., Dagallier, L.-P.M.J., Crozier, F., Ghogue, J.-P.,
Hoekstra, P.H., Kamdem, N.G., Johnson, D.M., Murray, N.A. & Sonké, B. (2022).
Flora of Cameroon – Annonaceae Vol 45. *PhytoKeys* 207: 1–532.
https://doi.org/10.3897/phytokeys.207.61432

[9] Johnson, D.M. & Murray, N.A. (2018). A revision of *Xylopia* L.
(Annonaceae): the species of Tropical Africa. *PhytoKeys* 97: 1–252.
https://doi.org/10.3897/phytokeys.97.20975

[10] Hoekstra, P.H., Wieringa, J.J., Maas, P.J.M., et al. (2021). Revision of
the African species of *Monanthotaxis* (Annonaceae). *Blumea* 66(2): 107–221.
https://doi.org/10.3767/blumea.2021.66.02.01
