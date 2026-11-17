"""Seed the dashboard with real pipeline runs over a handful of sample posts.

Every row is produced by actually running the encoders and the fusion network -
nothing here fabricates a probability or a label.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src import benchmark, db, pipeline  # noqa: E402

SAMPLES = [
    ("weather", 3,
     "Heavy rainfall warning issued for coastal Tamil Nadu over the next 48 hours as a "
     "low-pressure area forms over the Bay of Bengal.",
     "District weather bulletin", "X (Twitter)", "weather_alert"),
    ("weather", 11,
     "BREAKING: The government is using hidden cloud machines to control the weather in every "
     "district, share this before it is deleted!!",
     "Forwarded message", "WhatsApp", "weather_misinfo"),
    ("health", 5,
     "A measles vaccination drive for children under five will begin in government schools "
     "from Monday, the district health office said.",
     "Health department release", "Facebook", "health_alert"),
    ("health", 9,
     "SHOCKING: One spoon of this kitchen powder cures diabetes in three days, doctors hate it. "
     "Forward to ten friends now!",
     "Unknown blog", "WhatsApp", "health_misinfo"),
    ("technology", 2,
     "The company said its quarterly revenue rose 12 percent, driven by strong demand for its "
     "cloud services.",
     "Business daily", "News website", "tech_news"),
    ("technology", 7,
     "BREAKING: A hidden chip inside every phone transmits your conversations to a secret "
     "agency, leaked documents prove it!!",
     "Anonymous channel", "Telegram", "tech_misinfo"),
    ("politics", 6,
     "The election commission announced that voting in the state will be held in three phases "
     "starting next month.",
     "Election commission", "X (Twitter)", "politics_alert"),
    ("politics", 4,
     "EXPOSED: Voting machines are secretly programmed to flip every vote, share before this "
     "is removed!",
     "Viral post", "Facebook", "politics_misinfo"),
    ("disaster", 1,
     "A 4.8 magnitude earthquake was recorded off the Andaman coast, the national seismology "
     "centre said on Thursday.",
     "Seismology centre", "News website", "disaster_report"),
    ("disaster", 8,
     "BREAKING: A tsunami will hit the coast tonight at 9 pm, share this with everyone "
     "immediately!!!",
     "Chain message", "WhatsApp", "disaster_misinfo"),
    ("sports", 0,
     "The home team won the final by four wickets with two balls remaining at the Chinnaswamy "
     "stadium.",
     "Sports desk", "X (Twitter)", "sports_report"),
    ("sports", 12,
     "SHOCKING: The star player has been banned for life for match fixing, share before the "
     "news is removed!!",
     "Fan page", "Instagram", "sports_rumour"),
]


def _render(topic: str, variant: int) -> Image.Image:
    img = benchmark.render_scene(topic, variant)
    return img


def seed(limit: int = 12, explain_tokens: bool = False) -> int:
    pipeline.warm_up()
    created = 0
    for topic, variant, text, source, platform, slug in SAMPLES[:limit]:
        image = _render(topic, variant)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        raw = buffer.getvalue()

        config_upload_dir = Path(__import__("config").UPLOAD_DIR)
        filename = f"seed_{slug}.jpg"
        (config_upload_dir / filename).write_bytes(raw)

        meta = {
            "filename": f"{slug}.jpg",
            "stored_name": filename,
            "sha256": pipeline.hash_bytes(raw),
            "url": f"/media/{filename}",
            "mime": "image/jpeg",
        }
        result = pipeline.analyze(text=text, image=image, source=source,
                                  platform=platform, image_meta=meta,
                                  explain_tokens=explain_tokens)
        db.insert(result)
        created += 1
        print(f"  seeded [{created}] {result['label']:<10} "
              f"{result['confidence'] * 100:5.1f}%  {text[:58]}...")
    print(f"Seeded {created} analyses.")
    return created


if __name__ == "__main__":
    db.init_db()
    if db.count() > 0:
        print(f"Database already has {db.count()} records; nothing to do.")
    else:
        seed()
