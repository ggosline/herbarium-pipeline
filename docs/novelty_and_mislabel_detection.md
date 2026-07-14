# Detecting novel taxa and mis-identified specimens in a herbarium classifier

Draft methods and results, written for reuse in a paper. Numbers are from the
African Rubiaceae run of 2026-07-13/14 unless stated otherwise. Sections marked
**[not yet measured]** describe method and rationale only; do not report them as
results.

---

## 1. Motivation

A softmax classifier distributes all its probability mass across the classes it
was trained on. A specimen of a taxon the model has never seen therefore cannot
be rejected — it must be filed somewhere, and often is filed confidently. For a
curatorial tool this is the central failure mode: the model is most dangerous
precisely where it is most useful, because a confident wrong identification is
harder to catch than an uncertain one.

Two questions follow, and conflating them has cost us time before:

1. **Is this specimen a taxon the model has never seen?** (novelty / OOD)
2. **Is this specimen's existing label wrong?** (mis-identification)

They are not the same problem, they need different machinery, and — as we show
in §5 — one of them is *impossible* to answer for training specimens using the
model's own predictions, for reasons that are not obvious.

---

## 2. Model and data

* **Backbone**: ViT-Large, DINOv3 pretraining (`vit_large_patch16_dinov3.lvd1689m`),
  fine-tuned at 640 px.
* **Heads**: hierarchical — species (1,341 classes), genus (164), family (1) —
  trained jointly with loss weights 1.0 / 0.5 / 0.1. Geographic coordinates are
  sphere-encoded and fused with the image features before the heads (64-d MLP).
* **Data**: 99,857 African Rubiaceae specimen images from GBIF; 93,620 carry a
  determination, 6,237 are *indet*.
* **Class construction**: taxa with fewer than 20 images are dropped as
  unlearnable (`--sparse-threshold 20`); classes are capped at 500 images
  (`--max-per-class 500`), sampled round-robin across the rank below to preserve
  morphological breadth. No loss reweighting (`--class-weight-beta 0`; see §7).
* **Split**: 80/20 stratified by species, seed 42 → **68,252 train / 17,064
  held-out**.
* **Performance**: held-out top-1 **88.7 % species**, **97.3 % genus**.
  Calibration temperature fitted post hoc (Guo et al. 2017): T = 0.585.

---

## 3. A free, labelled novelty benchmark

Detecting novelty is usually evaluated by borrowing an unrelated dataset as the
"novel" class. That is a poor proxy for the curatorial reality, where the novel
taxon is a *congener* sitting on the same herbarium sheet stock.

The sparse-class filter provides a better benchmark at zero cost. Taxa with too
few images to learn are **excluded from training but retained in the metadata**,
so their specimens are, by construction, known-novel and known-labelled. For this
run the filter excluded 1,294 species, yielding two disjoint evaluation sets:

| set | specimens | taxa | what it tests |
|---|---:|---:|---|
| **A. Novel genus** | 651 | 87 genera | genuine out-of-distribution |
| **B. Novel species, known genus** | 7,653 | 1,136 species | *near*-OOD: a new species of a genus the model knows well |

Set B is the case that matters most for African Rubiaceae, and it is
fundamentally harder. **A new *Psychotria* IS a *Psychotria*.** The genus head is
*right* to be confident; only the species head has nowhere to put it. No
genus-level signal can detect set B even in principle — a prediction we test and
confirm in §4.3.

Indets are scored (they are exactly whom a curator wants triaged) but **excluded
from evaluation**: they have no true determination, so they can be called neither
novel nor in-distribution. This is not a technicality. Treating them as novel —
which a naive lookup does, since a missing species belongs to no known genus —
silently adds 6,237 ordinary specimens of *known* genera to the novel set and
drives max-softmax AUROC from 0.889 down to 0.694.

### 3.1 The negatives must be held-out, not merely in-distribution

The in-distribution comparison set must exclude the model's own training images.
This is the single most important methodological point in this section, and the
easiest to get wrong.

| specimens | n | top-1 accuracy | mean top-1 confidence |
|---|---:|---:|---:|
| train (seen during fitting) | 68,252 | **100.0 %** | **0.9985** |
| held-out | 17,064 | 88.7 % | 0.8870 |
| novel genus | 651 | — | 0.4015 |
| novel species, known genus | 7,653 | — | 0.5075 |

The model has effectively memorised its training set. Scoring novelty against
*all* in-distribution specimens — 80 % of which are memorised — therefore
inflates every result:

| negatives used | max-softmax AUROC, novel genus |
|---|---:|
| all in-distribution specimens | 0.964 |
| **held-out specimens only** | **0.889** |

Same model, same score, **8 points of fiction**. All results below use held-out
negatives exclusively. To make this reproducible rather than a matter of trust,
the train/validation split is now embedded in the checkpoint itself, and the
inference step emits a `split` column (`train` / `held-out` / `excluded`)
alongside every prediction.

---

## 4. Novelty scores

All scores are computed in a single forward pass, retaining both the logits and
the penultimate (1024-d) image embedding. Higher = more novel.

| score | definition | reference |
|---|---|---|
| **MSP** | 1 − max softmax | Hendrycks & Gimpel 2017 |
| **Energy** | −logsumexp(logits) | Liu et al. 2020 |
| **Mahalanobis** | distance to nearest class centroid, tied covariance | Lee et al. 2018 |
| **RMD** | Mahalanobis − distance to a single background Gaussian | Ren et al. 2021 |
| **k-NN** | cosine distance to the k-th nearest training specimen (k = 10) | Sun et al. 2022 |
| **1 − genus_conf** | 1 − max softmax of the *trained genus head* | this work |

Centroids and covariances are fitted on the **training split only**. Embeddings
are the image features alone, deliberately excluding the fused geographic vector:
a specimen collected in an unusual locality is not *morphologically* novel, and
fusing geography here would conflate the two.

### 4.1 Results

AUROC against held-out in-distribution negatives. "Flag rate" is the fraction of
the reviewed collection a curator must examine to recover 80 % of the novel
specimens — the operationally meaningful quantity.

| score | A. novel genus | flag @ 80 % | B. novel species, known genus | flag @ 80 % |
|---|---:|---:|---:|---:|
| **MSP + RMD** | **0.900** | **13.1 %** | **0.871** | 38.6 % |
| RMD | 0.894 | 15.1 % | 0.847 | 41.8 % |
| MSP | 0.889 | 17.8 % | 0.869 | 39.1 % |
| 1 − genus_conf | 0.887 | 20.1 % | 0.672 | 69.2 % |
| Mahalanobis | 0.822 | 30.7 % | 0.775 | 50.8 % |
| k-NN | 0.771 | 41.0 % | 0.714 | 59.6 % |
| Energy | 0.715 | 46.3 % | 0.620 | 72.2 % |

### 4.2 Plain Mahalanobis loses to max-softmax — because this is near-OOD

The OOD literature reports embedding-distance methods comfortably beating
max-softmax, typically lifting AUROC from ~0.88 to >0.95. **We observe the
opposite**: Mahalanobis (0.822) and k-NN (0.771) both fall well *below* MSP
(0.889).

We attribute this to the near-OOD structure of the problem. Published benchmarks
are overwhelmingly *far*-OOD (e.g. CIFAR-10 vs SVHN — different visual worlds).
Here **every novel taxon is another Rubiaceae**, photographed as a pressed sheet
on the same stock by the same institutions. The dominant directions of the
embedding space encode "is a mounted herbarium specimen" and "is a Rubiaceae" —
variation shared by novel and known taxa alike — and raw distance-to-centroid is
governed by that shared component rather than by the taxon-specific residual.

The prediction that follows is testable: removing the shared component should
recover the signal. **Relative Mahalanobis** (Ren et al. 2021) does exactly this,
subtracting the distance to a single class-agnostic Gaussian fitted over all
training features. It lifts the score from **0.822 → 0.894** (+7.2 points),
overtaking max-softmax, and the ensemble MSP + RMD reaches **0.900** while cutting
the curator's review burden from 17.8 % to **13.1 %** of the collection.

That a correction designed specifically for near-OOD recovers exactly the deficit
we predicted is, we argue, good evidence for the diagnosis rather than a lucky
hyperparameter.

### 4.3 Two negative results, reported deliberately

**Hierarchical disagreement is not useful.** The model carries a separately
trained genus head, so one can ask whether it agrees with the genus implied by
the species head. Disagreement is botanically interpretable and costs nothing,
and we expected it to help. It does not: AUROC **0.741** (set A) and **0.613**
(set B), far below plain max-softmax. We abandoned it.

**Genus confidence behaves exactly as theory demands** — which is the clearest
validation of the set A / set B decomposition. `1 − genus_conf` is the *strongest
single* signal for a novel **genus** (0.887, essentially tied with MSP) and is
close to *useless* for a novel **species within a known genus** (0.672). In the
latter case the genus really is known, and the model is right to be confident
about it. A method cannot detect what is not, at that rank, anomalous.

**Set B remains hard.** Even the best score requires flagging ~39 % of the
collection to recover 80 % of novel species. This is not yet a triage tool, and
we do not present it as one.

### 4.4 Confidence must be calibrated before any of this means anything

An earlier model of the same data was trained with inverse-frequency class
weights (w ∝ 1/n). This is a common reflex on long-tailed data and it is a trap:
weighted cross-entropy makes the network learn p(c|x) ∝ w_c · p_true(c|x), so the
*spread* of the weights acts as a raw logit bias at prediction time. With a
2,210× spread, genera represented by five specimens out-predicted *Psychotria*,
the commonest genus, whose recall collapsed to 11.2 % — while overall accuracy
still read 87.2 %, which is what hid it.

Max-softmax novelty detection on that model scored **AUROC 0.751**. This number
is meaningless: median top-1 confidence was 0.18, i.e. the confidence signal that
every score in §4 is built on had been destroyed. Removing the weighting
(`beta = 0`) and controlling the tail at the data level instead (sparse floor +
per-class cap) restored Psychotria to **99.7 %** held-out recall and took novelty
detection from 0.751 to 0.889 **with no change to the OOD method at all**.

The general point is worth stating plainly for a methods paper: *novelty
detection is downstream of calibration.* Any long-tail intervention that distorts
the logits will silently destroy the OOD signal, and overall accuracy will not
warn you.

---

## 5. Mis-identified specimens **in the training set**

### 5.1 The problem, and why the obvious approach cannot work

For held-out specimens, mis-identification is easy to surface: flag disagreement
between the model's prediction and the recorded determination.

For training specimens this **provably yields nothing**. Our model scores
**100.0 %** on its training set. Not 99.9 % — every single one of the 68,252
training specimens is "correctly" classified, so a prediction-vs-label
disagreement flag fires exactly zero times.

The reason is memorisation, and it is worse than merely unhelpful. If a sheet is
labelled *Psychotria* but is really *Chassalia*, the network has ample capacity
(ViT-Large, ~300 M parameters, 68 k images, 10 epochs) to memorise *this
particular image → Psychotria*. It **learns the error**, and then confidently
confirms the error back to the curator. Mis-labelled examples are precisely the
examples a network is *forced* to memorise, because they cannot be got right by
generalising.

That last observation is not just the problem — it is the solution.

### 5.2 Area Under the Margin (AUM)

Following Pleiss et al. (2020), we exploit *when* an example is learned rather
than *whether* it is. Networks fit clean, generalisable structure first and
memorise noise last (Arpit et al. 2017). Define, for training example *i* at
epoch *t*:

```
margin_i(t) = z_y(i)(t) − max_{c ≠ y(i)} z_c(t)
```

the logit assigned to its recorded label minus the largest competing logit.
AUM_i is this margin averaged over training epochs.

* A **correctly labelled** specimen is *supported* by every other specimen of its
  class. Its margin goes positive early and stays there. High AUM.
* A **mis-labelled** specimen is *opposed* by every correctly labelled specimen of
  the class it has been assigned to. The only way the network can fit it is to
  memorise it individually, which happens late. Its margin is low or negative for
  many epochs first. **Low or negative AUM.**

The averaging is essential and is the whole trick. **A final-epoch snapshot cannot
work** — by the last epoch memorisation has won, train accuracy is 100 %, and a
mis-labelled specimen looks exactly like a clean one. The signal lives in the
*history* of the margin, not its endpoint.

### 5.3 Cost and non-interference

AUM is **purely observational**: the margin is computed from logits the forward
pass already produces, under `no_grad`, and contributes no term to the loss and no
gradient. The trained model is bit-for-bit what it would have been. Storage is one
float and one counter per training image (~0.5 MB here); compute is one `gather`,
one masked `max` and one `index_add` per batch. We therefore enable it by default.

The per-specimen mean margin is embedded in the checkpoint, so the ranking travels
with the model that produced it.

### 5.4 Proposed validation **[not yet measured]**

AUM is implemented but its discriminative power on this dataset is **not yet
measured**, and should not be reported until it is. The intended protocol is
label-noise injection: flip a random 1 % of training labels to a random other
class, retrain, and measure how well AUM ranks the planted errors (AUROC, and
precision@k for curator-relevant k). This converts "we hope this finds mis-IDs"
into a number, and it is the only honest way to quote a detection rate before a
botanist has adjudicated a real sample.

We note two expected limitations. (i) Planted uniform-random noise is *easier*
than real herbarium mis-identification, which is concentrated between
morphologically similar congeners — so the measured AUROC will be an upper bound.
(ii) AUM ranks training specimens only; held-out mis-IDs are already surfaced by
ordinary prediction–label disagreement.

### 5.5 Alternatives considered

* **Out-of-fold prediction (k-fold cross-validation)**, as used by confident
  learning (Northcutt et al. 2021), is the rigorous alternative: every specimen is
  predicted by a model that never saw it. It is strictly more defensible than AUM
  and we regard it as the gold standard. Its cost is k× training (≈ 5× here), which
  AUM avoids entirely by extracting the signal from a single run we were performing
  anyway.
* **Embedding-geometry heuristics** — distance from a specimen to its *own*
  labelled class centroid, or label disagreement among its k nearest neighbours —
  are free, since the embeddings are already computed for §4. But they are
  compromised by the very effect they are trying to detect: fine-tuning drags a
  memorised mis-labelled image *toward* its wrong class in feature space. They
  surface gross errors and miss subtle ones. We use them as a first pass, not as a
  result.

---

## 6. Reproducibility

* The train/validation split is embedded in every checkpoint and emitted as a
  `split` column in the predictions table, so no downstream analysis has to
  reconstruct it — reconstruction depends on the seed, the sparse threshold, the
  per-class cap, the label rank, *and* which image files happened to be present,
  and is silently wrong if any of these drift.
* Where an older checkpoint predates split recording, the split is reconstructed
  and then **verified** against the per-class counts stored in the checkpoint; the
  analysis aborts rather than proceed on an unverified split.
* Embeddings are cached, so re-tuning a score (k, shrinkage) costs seconds rather
  than a fresh GPU pass over 100 k images.

## 7. Threats to validity

* **Single taxon, single region.** All results are from African Rubiaceae. The
  near-OOD argument in §4.2 predicts that a *phylogenetically broader* training
  set (e.g. all angiosperm families) would show a larger gap between embedding
  distance and max-softmax, because novel taxa would be further from the training
  manifold. This is untested.
* **The novel sets are rare taxa by construction.** Set A and set B are defined by
  the sparse filter, so novel specimens are also, on average, specimens of taxa
  with few collections. If rarity correlates with image quality or with
  collection-era artefacts, some of the measured separability may be attributable
  to those confounds rather than to morphology. A cleaner test would hold out a
  random sample of *well-collected* genera entirely.
* **Set B is contaminated by taxonomic opinion.** Some "novel species in a known
  genus" are synonyms or recent segregates of trained species. Such a specimen is
  arguably *not* novel, and counting it as such depresses the measured AUROC.
* **AUM is unvalidated on this data** (§5.4).

## References

Arpit, D. et al. (2017). *A Closer Look at Memorization in Deep Networks.* ICML.

Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML.

Hendrycks, D. & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and
Out-of-Distribution Examples in Neural Networks.* ICLR.

Lee, K. et al. (2018). *A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS.

Liu, W. et al. (2020). *Energy-based Out-of-distribution Detection.* NeurIPS.

Menon, A. K. et al. (2021). *Long-tail Learning via Logit Adjustment.* ICLR.

Northcutt, C. et al. (2021). *Confident Learning: Estimating Uncertainty in
Dataset Labels.* JAIR.

Pleiss, G. et al. (2020). *Identifying Mislabeled Data using the Area Under the
Margin Ranking.* NeurIPS.

Ren, J. et al. (2021). *A Simple Fix to Mahalanobis Distance for Improving
Near-OOD Detection.* arXiv:2106.09022.

Sun, Y. et al. (2022). *Out-of-Distribution Detection with Deep Nearest
Neighbors.* ICML.
