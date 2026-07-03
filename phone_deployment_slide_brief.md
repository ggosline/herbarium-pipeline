# Slide brief — "Publish & Identify on a Phone"

Paste this into NotebookLM (or hand to a designer) to generate a polished slide
that matches the existing deck and adds the missing deployment story. The
generated `herbarium_pipeline_intro.pptx` already carries an equivalent slide
(#18); this brief is for the illustrated NotebookLM version.

**Style to match the rest of the deck:** cream graph-paper background, deep-teal
accents, botanical engraving motif, clean bordered cards, one strong caption.

---

**Title:** Publish & Identify on a Phone
**Subtitle:** Publish a trained model to the Hugging Face Hub — anyone identifies
specimens from a phone browser, no install.

**Central flow (left → right, three cards with arrows):**
1. `Trained checkpoint (.ckpt)`  — arrow labelled **push_model.py** →
2. `Hugging Face Hub — model repo`  (small print: `model.ckpt · nameslist.json · config.json`)  — arrow labelled **auto-discovered** →
3. `herbarium-id Space (Gradio, free CPU tier)`

**Left column — "Publishing a model":**
- `push_model.py` slims the checkpoint (drops optimizer state, ~⅓ the size) and uploads `model.ckpt` + `nameslist.json` + `config.json`.
- Repo is tagged `herbarium-pipeline`, so the Space discovers it automatically — tap ⟳ to refresh, no redeploy.
- `config.json` carries image size, label rank, and a **calibration temperature** so confidence is honest, not pinned near 100 %.
- One-time `huggingface-cli login` with a write token.

**Right column — "On a phone" (beside a simple phone mockup):**
- Open the Space URL in any phone browser — nothing to install.
- Pick a model, tap to photograph a sheet with the rear camera.
- Top-5 species / family + confidence in a second or two.
- Optional lat/lon sharpens geo-aware models.

**Bottom caption bar:**
Live now: https://huggingface.co/spaces/ggosline/herbarium-id
· models: africa-angiosperms-family · olacaceae-species · uvaria-species

---

**Accuracy notes (keep these correct if the tool rephrases):**
- The Space applies `softmax(logits / T)` using each model's `config.json`
  temperature; existing models were calibrated post-hoc (no retraining) via
  `calibrate_temperature.py`, new ones get a fitted `T` at publish time.
- Models are stored on the HF Hub and can be served from other scale-to-zero
  hosts (RunPod Serverless, Google Cloud Run, Modal) — the Space is just one
  front end.
