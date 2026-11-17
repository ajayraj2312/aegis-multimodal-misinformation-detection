"""Flask application: dashboard, multimodal analysis, history, model performance."""
from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from PIL import Image, UnidentifiedImageError

import config
from src import db, pipeline
from src.image_encoder import ImageEncoderUnavailable
from src.text_encoder import ModelUnavailable

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production"),
    JSON_SORT_KEYS=False,
)

PLATFORMS = ["X (Twitter)", "Facebook", "WhatsApp", "Instagram", "Telegram",
             "YouTube", "News website", "Other"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_metrics() -> dict | None:
    path = config.MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def models_ready() -> bool:
    return all((config.MODELS_DIR / f"{k}.pt").exists()
               for k in ("fusion", "text_only", "image_only"))


def validate_text(text: str) -> str | None:
    cleaned = (text or "").strip()
    if len(cleaned) < config.MIN_TEXT_CHARS:
        return f"Please enter at least {config.MIN_TEXT_CHARS} characters of post text."
    if len(cleaned) > config.MAX_TEXT_CHARS:
        return f"Post text must be under {config.MAX_TEXT_CHARS} characters."
    return None


def save_upload(file_storage) -> tuple[bytes, dict]:
    raw = file_storage.read()
    name = file_storage.filename or "upload"
    ext = Path(name).suffix.lower().lstrip(".")
    if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image type. Use PNG, JPG or JPEG.")
    if not raw:
        raise ValueError("The uploaded image is empty.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read the image file: {exc}") from exc

    filename = f"{uuid.uuid4().hex}.{ 'jpg' if ext == 'jpeg' else ext}"
    (config.UPLOAD_DIR / filename).write_bytes(raw)
    meta = {
        "filename": name,
        "stored_name": filename,
        "sha256": pipeline.hash_bytes(raw),
        "url": url_for("media", filename=filename),
        "mime": image.get_format_mimetype(),
    }
    return raw, {**meta, "_image": image}


def get_image_and_meta(raw: bytes, meta: dict):
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    stats_data = db.stats()
    return render_template(
        "dashboard.html",
        active="dashboard",
        stats=stats_data,
        recent=db.recent(8),
        trend=db.trend(14),
        platforms=db.platform_breakdown(),
        metrics=load_metrics(),
        models_ready=models_ready(),
    )


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        return render_template("analyze.html", active="analyze",
                               platforms=PLATFORMS, form={}, models_ready=models_ready())

    text = request.form.get("text", "")
    source = request.form.get("source", "")
    platform = request.form.get("platform", "")
    file = request.files.get("image")

    error = validate_text(text)
    if error:
        flash(error, "danger")
        return render_template("analyze.html", active="analyze", platforms=PLATFORMS,
                               form={"text": text, "source": source,
                                     "platform": platform},
                               models_ready=models_ready()), 400

    if not models_ready():
        flash("The models are not trained yet. Run `python -m src.train` and "
              "`python -m src.evaluate`, then restart the app.", "danger")
        return render_template("analyze.html", active="analyze", platforms=PLATFORMS,
                               form={"text": text, "source": source,
                                     "platform": platform}, models_ready=False), 503

    image, meta = None, None
    if file and file.filename:
        try:
            raw, meta = save_upload(file)
            image = get_image_and_meta(raw, meta)
        except ValueError as exc:
            flash(str(exc), "danger")
            return render_template("analyze.html", active="analyze", platforms=PLATFORMS,
                                   form={"text": text, "source": source,
                                         "platform": platform},
                                   models_ready=True), 400

    try:
        result = pipeline.analyze(
            text=text.strip(), image=image, source=source, platform=platform,
            image_meta={k: v for k, v in (meta or {}).items() if not k.startswith("_")},
        )
    except (ModelUnavailable, ImageEncoderUnavailable) as exc:
        flash(str(exc), "danger")
        return render_template("analyze.html", active="analyze", platforms=PLATFORMS,
                               form={"text": text, "source": source,
                                     "platform": platform}, models_ready=True), 503

    analysis_id = db.insert(result)
    return redirect(url_for("result", analysis_id=analysis_id))


@app.route("/result/<int:analysis_id>")
def result(analysis_id: int):
    record = db.get(analysis_id)
    if not record:
        abort(404)
    return render_template("result.html", active="history", r=record,
                           disclaimer=config.DISCLAIMER, metrics=load_metrics())


@app.route("/history")
def history():
    page = request.args.get("page", 1, type=int)
    listing = db.list_page(
        query=request.args.get("q", "").strip(),
        label=request.args.get("label", "").strip(),
        modality=request.args.get("modality", "").strip(),
        page=page, per_page=10,
    )
    return render_template("history.html", active="history", listing=listing,
                           stats=db.stats())


@app.route("/history/<int:analysis_id>/delete", methods=["POST"])
def delete_analysis(analysis_id: int):
    db.delete(analysis_id)
    flash("Record deleted.", "success")
    return redirect(request.referrer or url_for("history"))


@app.route("/performance")
def performance():
    return render_template("performance.html", active="performance",
                           metrics=load_metrics(), models_ready=models_ready())


@app.route("/methodology")
def methodology():
    return render_template("methodology.html", active="methodology",
                           metrics=load_metrics(), disclaimer=config.DISCLAIMER)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if not models_ready():
        return jsonify({"error": "models_not_trained"}), 503

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        source = payload.get("source", "")
        platform = payload.get("platform", "")
        file = None
    else:
        text = request.form.get("text", "")
        source = request.form.get("source", "")
        platform = request.form.get("platform", "")
        file = request.files.get("image")

    error = validate_text(text)
    if error:
        return jsonify({"error": "invalid_text", "message": error}), 400

    image, meta = None, None
    if file and file.filename:
        try:
            raw, meta = save_upload(file)
            image = get_image_and_meta(raw, meta)
        except ValueError as exc:
            return jsonify({"error": "invalid_image", "message": str(exc)}), 400

    result = pipeline.analyze(
        text=text.strip(), image=image, source=source, platform=platform,
        image_meta={k: v for k, v in (meta or {}).items() if not k.startswith("_")},
        explain_tokens=False,
    )
    result["id"] = db.insert(result)
    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    return jsonify({"stats": db.stats(), "trend": db.trend(14),
                    "platforms": db.platform_breakdown()})


@app.route("/api/history")
def api_history():
    return jsonify(db.list_page(
        query=request.args.get("q", ""), label=request.args.get("label", ""),
        modality=request.args.get("modality", ""),
        page=request.args.get("page", 1, type=int), per_page=20))


@app.route("/media/<path:filename>")
def media(filename: str):
    return send_from_directory(config.UPLOAD_DIR, filename)


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "models_trained": models_ready(),
        "database_records": db.count(),
        "text_encoder": config.TEXT_MODEL_NAME,
    })


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", active="", code=404,
                           message="That page does not exist."), 404


@app.errorhandler(413)
def too_large(_e):
    flash("That image is larger than the 8 MB limit.", "danger")
    return redirect(url_for("analyze"))


@app.errorhandler(500)
def server_error(_e):  # pragma: no cover
    return render_template("error.html", active="", code=500,
                           message="Something went wrong while processing the request."), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def seed_demo_if_empty() -> None:
    if not config.SEED_DEMO_ON_EMPTY or db.count() > 0 or not models_ready():
        return
    try:
        from scripts.seed_demo import seed
        seed(limit=12, explain_tokens=False)
    except Exception as exc:  # pragma: no cover - seeding must never block boot
        app.logger.warning("Demo seeding skipped: %s", exc)


db.init_db()
with app.app_context():
    seed_demo_if_empty()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
