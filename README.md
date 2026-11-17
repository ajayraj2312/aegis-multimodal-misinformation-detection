# AEGIS — AI-Based Misinformation Detection with Multimodal Learning

A working final-year engineering project prototype that classifies a social-media post as
**Credible** or **Misleading** by jointly analysing its **text** and its **image**, with a
cross-modal consistency check, an explanation of the signals that drove the decision, a
SQLite-backed analytics dashboard and a model-evaluation report computed on a held-out test
split.

Based on *"AI-Based Misinformation Detection with Multimodal Learning"* (Phase-I report,
Dept. of CSE, Sathyabama Institute of Science and Technology).

---

## 1. What actually runs

```
post text ─► clean / validate ─► DistilBERT (frozen) ─► 768-d contextual embedding ─┐
                                                                                     ├─► projection to shared 256-d space
post image ─► decode / resize / normalise ─► ResNet18 (frozen) ─► 512-d embedding ───┘        │
                                                                                              ├─► cosine similarity  ─┐
                                                                                              │   + consistency head  ─┤ cross-modal
                                                                                              │                        │   consistency
                                                                                              ▼                        │
                                                                          gated concat + t⊙i + |t−i|  (1024-d)         │
                                                                                              │                        │
                                                                                              ▼                        ▼
                                                                                    fusion MLP (256→64) ─► credibility head ─► Credible / Misleading
                                                                                                       ─► consistency head  ─► Consistent / Inconsistent
```

* **Text encoder** — `distilbert-base-uncased`, frozen, masked mean pooling → 768-d.
* **Image encoder** — `resnet18` (ImageNet weights), classifier head removed → 512-d.
* **Fusion network** — per-modality projectors, learned modality gates, explicit cross-modal
  interaction terms (element-wise product and absolute difference), 1024 → 256 → 64 MLP.
* **Cross-modal consistency** — cosine similarity in the shared space *plus* a dedicated
  consistency head, trained with a contrastive loss that pulls matching text/image pairs
  together and pushes mismatched pairs apart.
* **Explanations** — occlusion attribution (each word removed in turn, classifier re-run) and
  leave-one-modality-out ablation for the text/image contributions.

Nothing in the classification path is keyword matching; the cue vocabulary shown in the UI is
display-only.

## 2. Features

| Module | Where |
| --- | --- |
| Analytics dashboard (cards, donut, 14-day trend, platform bars, recent activity) — every number read from SQLite | `templates/dashboard.html`, `src/db.py` |
| Post analysis form (drag & drop, preview, validation, loading steps, text-only mode) | `templates/analyze.html` |
| Transformer text analysis panel (tokens, embedding stats, linguistic features, token influence) | `src/text_encoder.py`, `src/explain.py` |
| CNN image analysis panel (pixel statistics, artefact heuristics, embedding stats) | `src/image_encoder.py`, `src/image_stats.py` |
| Multimodal fusion + cross-modal consistency | `src/fusion_model.py`, `src/consistency.py` |
| Credibility classification with confidence and contributions | `src/pipeline.py` |
| Model performance page (accuracy, precision, recall, F1, AUC, confusion matrices, ROC, training curves) | `src/evaluate.py`, `templates/performance.html` |
| Prediction history (search, filter by result/modality, pagination, detail view, delete) | `src/db.py`, `templates/history.html` |

## 3. Tech stack

Python 3.10 · Flask · PyTorch · torchvision · Hugging Face Transformers · NumPy · Pandas ·
scikit-learn · Pillow · SQLite · Bootstrap 5 · Chart.js · Gunicorn.

## 4. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# one-off: build the benchmark features, train the three heads, evaluate them
python -m src.train        # ~90 s on a laptop CPU (downloads DistilBERT + ResNet18 once)
python -m src.evaluate     # writes models/metrics.json

python app.py              # http://127.0.0.1:5000
```

The app boots with an empty dashboard. It seeds itself with 12 sample analyses on first run
(`SEED_DEMO_ON_EMPTY=1`, set to `0` to disable) — those rows are real pipeline runs over sample
posts, not invented numbers. You can also run `python scripts/seed_demo.py` manually.

## 5. The benchmark and the numbers on `/performance`

`src/benchmark.py` builds **MPB-1 (Multimodal Proxy Benchmark)**: 120 curated claim texts
(72 credible / 48 misleading, six topics) expanded into **432 labelled text–image pairs**.

* Matching pairs (image topic = claim topic) keep the claim's own label.
* Mismatched pairs (image topic ≠ claim topic) are labelled **misleading**, because an image
  that does not show what the text claims is itself a misinformation signal.
* A share of images carries synthetic editing artefacts (caption overlay, spliced patch, heavy
  re-compression), more often on posts with misleading text.
* To keep the benchmark from being trivially separable, 50% of mismatched pairs use a
  *visually adjacent* distractor (a storm scene for a weather claim rather than an unrelated
  beach), 12% of images are heavily degraded, and 22% of mismatched-but-credible-text pairs keep
  the credible label (an illustrative/stock image is not itself misinformation). These are
  documented benchmark properties, not tuning knobs for the score.

Every metric on the performance page is computed by `src/evaluate.py` on the **held-out 15%
test split** (64 pairs), with the decision threshold chosen on the validation split. **No number
is hard-coded.**

### Measured results (seed 42, 64 held-out pairs)

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Text-only (DistilBERT) | 0.891 | 0.960 | 0.800 | 0.873 | 0.813 |
| Image-only (ResNet18) | 0.500 | 0.484 | 1.000 | 0.652 | 0.703 |
| **Multimodal fusion** | **0.953** | **0.935** | **0.967** | **0.951** | **0.957** |

Cross-modal consistency head: 0.938 accuracy, 0.912 AUC; mean cosine 0.775 for consistent vs
−0.130 for inconsistent pairs. The image-only model is deliberately kept in the table even
though it is weak — a single image says little about whether a claim is true, which is exactly
why it is used as a *consistency reference* rather than a standalone detector. All of these
numbers are regenerated by `python -m src.train && python -m src.evaluate`.

**Read this before quoting the results.** MPB-1 is a proxy corpus with procedurally rendered
scene images, and it is small (64 held-out pairs), so the confidence interval on every metric
above is wide. It demonstrates that the pipeline learns, that the fusion model handles the
mismatched-image case better than a text-only model, and that the consistency head is
measurable — it does **not** establish real-world accuracy or comparability with Fakeddit/Weibo
benchmarks. To train on a real corpus, replace `src/benchmark.py` with a loader that returns the
same sample dicts (`text, text_topic, image_topic, label, consistency, image_path`) and re-run
the two commands; nothing else changes.

## 6. HTTP API

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -F "text=Heavy rainfall warning issued for coastal Tamil Nadu" \
  -F "platform=X (Twitter)" \
  -F "image=@photo.jpg"
```

`GET /healthz`, `GET /api/stats`, `GET /api/history` are also available. `/api/analyze` returns
the full result dict (label, confidence, per-modality scores, contributions, consistency,
explanation) and stores the record.

## 7. Deploying to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ajayraj2312/aegis-multimodal-misinformation-detection)

The repository ships a `render.yaml` Blueprint, `build.sh` (installs CPU-only PyTorch on Linux)
and a `Procfile`.

1. Push this repository to GitHub (public).
2. Click the **Deploy to Render** button above, or on Render: **New → Blueprint**, pick the
   repository, then approve the service it proposes. `render.yaml` supplies the build command,
   start command, health check and every environment variable, so there is nothing to fill in.
3. Wait for the build (≈6–10 min: the CPU PyTorch wheel plus the model weight downloads) and
   open the service URL. The health check is `/healthz`.

Notes for the free tier (`plan: free`, 512 MB RAM):

* The blueprint sets `MISINFO_QUANTIZE=1`, which applies int8 dynamic quantisation to the
  transformer at load time to keep it inside 512 MB. If the service still restarts with an
  out-of-memory error, switch the plan to **Starter (2 GB)** — no code change needed.
* The filesystem is **ephemeral** on the free tier: SQLite records and uploaded images are lost
  on every redeploy/restart. Attach a Render **persistent disk** (e.g. mounted at `/data`) and
  set `DATABASE_PATH=/data/misinfo.db`, `INSTANCE_DIR=/data` to keep history.
* First request after a boot loads the models (a few seconds). `build.sh` pre-downloads the
  weights during the build, so this is CPU warm-up, not a network download.

## 8. Project layout

```
app.py                  Flask routes (pages + JSON API)
config.py               paths, model names, limits, disclaimer
src/
  benchmark.py          MPB-1 dataset: curated claims + procedural scene renderer
  features.py           cached encoder pass over the benchmark (artifacts/features.npz)
  text_encoder.py       DistilBERT encoder + tokenizer
  image_encoder.py      ResNet18 encoder
  image_stats.py        pixel statistics and artefact heuristics
  fusion_model.py       FusionNet, TextOnlyNet, ImageOnlyNet
  consistency.py        cross-modal consistency scoring + calibration
  explain.py            linguistic features, occlusion attribution, explanation text
  pipeline.py           end-to-end inference and modality ablation
  db.py                 SQLite schema, inserts, aggregates, history queries
  train.py              stratified split, training loop, checkpointing
  evaluate.py           held-out metrics -> models/metrics.json
scripts/seed_demo.py    real pipeline runs over sample posts
templates/ static/      UI (Bootstrap 5 + Chart.js)
models/                 checkpoints + metrics.json (generated by train/evaluate)
```

## 9. Limitations

1. **It is an automated credibility assessment, not a fact-checker.** A confidence score is the
   model's belief, not evidence that a claim is true.
2. **No manipulation forensics.** ResNet18 is a generic backbone; the UI's "visual artefact
   indicators" are transparent pixel statistics (overlay-band edge density, JPEG quantisation
   estimate, blockiness), clearly labelled as heuristics.
3. **Frozen encoders.** Fine-tuning DistilBERT and ResNet18 on a large real multimodal corpus is
   the natural next step and will change the metrics substantially.
4. **English-only text**, one image per post, no propagation or account-level social signals.

## 10. Reproducibility

All training uses `seed=42` (data generation, split, weight init, batch order). The split is
written to `models/split.json` and the full report to `models/metrics.json`; `/performance`
renders that report verbatim.
