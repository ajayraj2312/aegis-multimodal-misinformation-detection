"""A small, reproducible multimodal benchmark used to train and evaluate the models.

The benchmark pairs curated claim texts (labelled credible / misleading) with
scene images rendered procedurally from six topic categories. It is a *proxy*
benchmark, not a real-world fact-check corpus; its purpose is to give every
reported number an honest held-out test split that anyone can regenerate with
`python -m src.train`.

Pair labelling rule
-------------------
label = 1 (misleading)  if the claim text is misleading OR the image topic does
                        not match the topic the text talks about
label = 0 (credible)    otherwise

A fraction of images carry synthetic editing artefacts (caption overlays,
spliced patches, heavy JPEG re-compression). Artefacts are more frequent on
posts with misleading text, mirroring how edited visuals are distributed in
real misinformation, which gives the image-only baseline a weak but genuine
signal.

Consistency label: 1 when the image topic matches the text topic, else 0.
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TOPICS: List[str] = ["weather", "health", "technology", "politics", "disaster", "sports"]

CREDIBLE_CLAIMS: Dict[str, List[str]] = {
    "weather": [
        "The India Meteorological Department has issued a heavy rainfall warning for coastal Tamil Nadu over the next 48 hours.",
        "Cyclone Dana is expected to make landfall near the Odisha coast late on Thursday night, the weather office said.",
        "Chennai recorded 12 cm of rain in the 24 hours ending 8:30 am on Monday, the highest this month.",
        "A yellow alert has been issued for six districts as a low-pressure area forms over the Bay of Bengal.",
        "Temperatures in Delhi are forecast to stay above 40 degrees Celsius until the weekend, officials said.",
        "The monsoon withdrawal is running about a week behind the normal schedule this year, IMD data shows.",
        "Light to moderate rain is likely over Kerala for the next three days, according to the regional bulletin.",
        "Schools in three coastal districts will remain closed on Tuesday after the weather department's red alert.",
        "A heatwave declaration was issued for two districts after temperatures crossed 45 degrees Celsius on Wednesday.",
        "Snowfall in the higher reaches of Himachal Pradesh has closed the Rohtang Pass road, local authorities confirmed.",
        "The weather station at Meenambakkam logged 9.4 cm of rainfall between Sunday night and Monday morning.",
        "Fishermen have been advised not to venture into the sea until the deep depression weakens over the Arabian Sea.",
    ],
    "health": [
        "The health department has reported 46 new cases of dengue in the district this week, officials said.",
        "A measles vaccination drive for children under five will begin in government schools from Monday.",
        "Researchers found that the new drug reduced hospital admissions by 18 percent in a randomised trial.",
        "The hospital has added 50 new beds to its intensive care unit, according to a statement released on Tuesday.",
        "Doctors recommend a balanced diet and regular exercise rather than supplements for most people.",
        "The vaccination centre at the district hospital will stay open until 8 pm on weekdays this month.",
        "State health officials confirmed that the fever outbreak is under control with no new cases in the last week.",
        "A peer-reviewed study of 4,200 patients found the treatment was safe over a two-year follow-up.",
        "The national drug regulator has approved the generic version after reviewing the clinical trial data.",
        "Health workers will screen blood pressure for adults over 30 at all primary health centres this month.",
        "People with chronic conditions are advised to continue their medication and consult a physician before stopping.",
        "The District Medical Officer said the water samples tested negative for contamination on Friday.",
    ],
    "technology": [
        "The company said its quarterly revenue rose 12 percent, driven by strong demand for its cloud services.",
        "A software update released on Monday fixes the login issue reported by some users last week.",
        "Researchers published a paper describing a new compression method that reduces file size by a third.",
        "The telecom regulator published its consultation paper on spectrum allocation for the next auction.",
        "Engineers at the institute demonstrated a prototype battery that charges in under twenty minutes.",
        "The smartphone maker said the device will be available in stores from the first week of next month.",
        "The agency warned users about a phishing campaign that mimics its login page and asks for OTPs.",
        "The open-source project released version 4.2 with better documentation and a smaller install size.",
        "The data centre operator said the new facility will run entirely on renewable power from next year.",
        "Security researchers reported the vulnerability to the vendor, which patched it within 48 hours.",
        "The ministry announced a chip manufacturing incentive scheme covering design and fabrication units.",
        "Independent tests found the laptop battery lasted nearly eleven hours under mixed use.",
    ],
    "politics": [
        "The election commission announced that voting in the state will be held in three phases starting next month.",
        "Parliament passed the bill after a debate that lasted more than six hours on Tuesday.",
        "The chief minister said the budget allocation for rural roads will increase by nine percent this year.",
        "The state cabinet approved a proposal to recruit 3,500 teachers in government schools.",
        "The opposition party released its manifesto, promising a review of the power tariff structure.",
        "The minister told reporters that the inquiry committee will submit its report within four weeks.",
        "Voter turnout in the by-election was recorded at 68 percent, the commission's preliminary data shows.",
        "The municipal corporation announced that property tax payments can now be made online.",
        "The governor returned the ordinance to the assembly with a request for clarification on two clauses.",
        "Officials said the new ward boundaries will take effect after the gazette notification is published.",
        "The committee recommended that the free bus scheme be evaluated before it is expanded statewide.",
        "The election commissioner said the model code of conduct applies from the date of the announcement.",
    ],
    "disaster": [
        "A 4.8 magnitude earthquake was recorded off the Andaman coast, the national seismology centre said.",
        "Rescue teams recovered six people from the collapsed building in the early hours of Wednesday.",
        "The district administration opened 14 relief camps after the river crossed the danger mark.",
        "Fire services brought the warehouse blaze under control after four hours, officials said.",
        "The state requested two additional battalions of disaster response personnel from the centre.",
        "Flood water has receded in most wards, though low-lying areas remain submerged, the corporation said.",
        "The meteorological centre attributed the deluge to a low pressure system that stalled over the coast.",
        "Officials confirmed that the bridge was closed for repairs after engineers found cracks in the piers.",
        "Relief material including rice and drinking water was distributed in 22 villages on Thursday.",
        "The report said 340 houses were damaged and the assessment for compensation is under way.",
        "Power supply was restored in most parts of the city after the substation was repaired.",
        "Volunteers from the local college helped clear debris from the school building on Saturday.",
    ],
    "sports": [
        "The home team won the final by four wickets with two balls remaining at the Chinnaswamy stadium.",
        "The captain said the squad will be announced after the selection committee meeting on Friday.",
        "The opening batsman scored 118 runs off 96 balls, his third century of the season.",
        "The football club confirmed that its midfielder will miss the next two matches with a hamstring injury.",
        "The tournament will be played across six venues between March and May, the board announced.",
        "The athlete qualified for the final with a timing of 10.24 seconds in the heats.",
        "The coach told reporters that the team will focus on fielding and fitness during the training camp.",
        "The umpire's decision was referred to the third umpire, who confirmed the batsman was out.",
        "Ticket sales for the second test match opened on Monday, the state association said.",
        "The badminton player lost the semi-final in three games after a 74-minute contest.",
        "The league table shows the club in fourth place with two matches still to play.",
        "The sports ministry increased the annual grant for the national training academy by 8 percent.",
    ],
}

MISLEADING_CLAIMS: Dict[str, List[str]] = {
    "weather": [
        "BREAKING: The government secretly controls the weather with hidden cloud machines installed in every district.",
        "SHOCKING: Scientists confirm that drinking hot water before bed prevents all viral infections permanently.",
        "Miracle cloud-seeding towers will bring rain to every drought area within 24 hours, share before it is deleted.",
        "EXPOSED: Weather apps are deliberately showing fake temperatures so that power companies can profit.",
        "Forward this to 10 friends and rain will reach your village this week, the department has confirmed.",
        "A secret plan to create artificial snowfall in Chennai using refrigerated aircraft will start next month.",
        "Doctors hate it: one tablespoon of this powder makes you immune to every seasonal flu overnight.",
        "The real reason for the flood is a hidden weather machine that the government does not want you to know about.",
    ],
    "health": [
        "SHOCKING: One spoon of this kitchen powder cures diabetes in three days, doctors hate it.",
        "The new vaccine contains a microchip that silently tracks every person who receives it.",
        "Miracle herb sold only on one website removes 100 percent of toxins from the body overnight.",
        "Forward this message to five friends, a child will die tonight if you ignore it, the hospital confirms.",
        "BREAKING: All medicines for blood pressure are being banned because they cause brain damage.",
        "Doctors are hiding the fact that lemon water kills every virus, including the one in your tap water.",
        "This viral home remedy has been proven to reverse paralysis in 24 hours, according to a leaked report.",
        "EXPERTS SAY: never take antibiotics again, the body can defeat every infection by itself.",
    ],
    "technology": [
        "BREAKING: A hidden chip inside every phone transmits your conversations to a secret agency.",
        "SHOCKING: WhatsApp will start charging 500 rupees per month from tomorrow, forward to all your groups.",
        "This free app doubles your phone battery instantly and unlocks unlimited data for life.",
        "Scientists have built an AI that predicts your death date exactly, and it is already in your phone.",
        "The government will delete all bank accounts linked to phones without this setting turned on tonight.",
        "Miracle USB stick cleans every virus from your computer and boosts internet speed by 500 percent.",
        "Leaked report: all smartphones will be shut down permanently on Sunday for a secret upgrade.",
        "One click on this link gives you 10,000 free followers, no login required, share to ten friends.",
    ],
    "politics": [
        "BREAKING: Voting machines are secretly programmed to flip every vote, leaked documents prove it.",
        "SHOCKING: The government has banned cash withdrawals from all banks starting Sunday midnight.",
        "Forward this: a new law will cancel the pension of everyone who does not link their phone this week.",
        "EXPOSED: The election will be cancelled and the army will take over the country next month.",
        "The chief minister has secretly signed away all water rights to a foreign company, share before removal.",
        "Miracle scheme gives every citizen 15 lakh rupees deposited directly, register with your Aadhaar now.",
        "A secret list of 20 lakh deleted voters was found and the media is hiding it from you.",
        "The parliament passed a law that takes away your land if you do not vote for the ruling party.",
    ],
    "disaster": [
        "BREAKING: A tsunami will hit the coast tonight at 9 pm, share this with everyone immediately.",
        "SHOCKING: The dam has already collapsed and the government is hiding the deaths, forward fast.",
        "A leaked map shows that half the city will sink into the ground before sunrise.",
        "Miracle: the flood water turned into salt because of a secret chemical experiment, see the video.",
        "The earthquake was caused by a secret underground test and the media will not tell you this.",
        "Forward to ten people: a gas leak in the city will kill everyone who sleeps before midnight.",
        "EXPOSED: All relief camps are fake, no food has been given to anyone since the flood.",
        "Scientists warn that a second tsunami, twice as big, is coming in the next 48 hours.",
    ],
    "sports": [
        "BREAKING: The star player has been banned for life for match fixing, share before the news is removed.",
        "SHOCKING: The world cup final was fixed and the umpire has confessed, forward to all fans.",
        "Miracle drink used by one team gives unlimited stamina, the board has banned it secretly.",
        "The captain has died in a road accident, prayers please, forward to ten groups.",
        "EXPOSED: The entire league is scripted and the results are decided by a TV channel.",
        "This free app gives you live matches for life if you share the link with five friends.",
        "The government has decided to cancel all cricket matches in the country from next week.",
        "Leaked audio proves that the coach paid the referee before the semi-final.",
    ],
}

ARTEFACT_OVERLAY_TEXT = "SHOCKING!! SHARE NOW"

# Visually / semantically adjacent topics. Half of the mismatched pairs draw their
# distractor image from here, which makes the consistency task non-trivial: a storm
# scene next to a weather claim is a much harder negative than a beach next to it.
ADJACENT_TOPICS: Dict[str, List[str]] = {
    "weather": ["disaster", "sports"],
    "health": ["technology", "sports"],
    "technology": ["health", "politics"],
    "politics": ["disaster", "sports"],
    "disaster": ["weather", "politics"],
    "sports": ["health", "politics"],
}
DEGRADED_RATE = 0.12   # share of images downscaled + blurred (weak visual evidence)

# A strict "mismatch => misleading" rule makes the benchmark trivially separable and
# produces degenerate 100% scores. Real posts frequently use illustrative, stock or
# generic imagery for a genuine claim, so a fraction of mismatched pairs keeps the
# claim's own (credible) label. This is documented annotation ambiguity, not tuning.
MISMATCH_ILLUSTRATIVE_RATE = 0.22


# ---------------------------------------------------------------------------
# Scene rendering
# ---------------------------------------------------------------------------
def _gradient(size: int, top, bottom, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(1, size - 1)
        colour = tuple(int(top[c] + (bottom[c] - top[c]) * t) for c in range(3))
        draw.line([(0, y), (size, y)], fill=colour)
    return img


def _scatter(draw: ImageDraw.ImageDraw, rng: random.Random, count: int, size: int,
             colour, radius_range=(1, 3)) -> None:
    for _ in range(count):
        x, y = rng.randrange(size), rng.randrange(size)
        r = rng.randint(*radius_range)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)


def render_scene(topic: str, variant: int, size: int = 224) -> Image.Image:
    """Render one deterministic scene image for a topic."""
    rng = random.Random(hash((topic, variant)) & 0xFFFFFFFF)
    shift = (variant * 17) % 40

    if topic == "weather":
        img = _gradient(size, (58 + shift, 66 + shift, 82), (176, 180, 186), rng)
        draw = ImageDraw.Draw(img)
        ground = size * 0.62
        draw.rectangle([0, ground, size, size], fill=(232, 236, 240))
        for i in range(5):
            x = rng.randrange(0, size)
            y = rng.randrange(0, int(ground))
            w = rng.randrange(size // 5, size // 2)
            draw.ellipse([x, y, x + w, y + w // 3], fill=(126, 132, 142))
        _scatter(draw, rng, 140, size, (250, 250, 252), (1, 2))
        img = img.filter(ImageFilter.GaussianBlur(0.6))

    elif topic == "health":
        img = _gradient(size, (232, 246, 244), (198, 226, 222), rng)
        draw = ImageDraw.Draw(img)
        cx, cy, arm = size // 2, int(size * 0.42), size // 8
        draw.rectangle([cx - arm, cy - arm * 2, cx + arm, cy + arm * 2], fill=(206, 60, 60))
        draw.rectangle([cx - arm * 2, cy - arm, cx + arm * 2, cy + arm], fill=(206, 60, 60))
        for i in range(3):
            x = 18 + i * (size // 4)
            draw.rounded_rectangle([x, size - 62, x + size // 7, size - 22],
                                   radius=10, fill=(255, 255, 255),
                                   outline=(120, 150, 150), width=2)
        _scatter(draw, rng, 40, size, (255, 255, 255, 0), (1, 2))

    elif topic == "technology":
        img = _gradient(size, (12, 18, 36), (30, 48, 78), rng)
        draw = ImageDraw.Draw(img)
        for _ in range(26):
            x1, y1 = rng.randrange(size), rng.randrange(size)
            x2, y2 = x1 + rng.randrange(-70, 70), y1 + rng.randrange(-70, 70)
            draw.line([x1, y1, x2, y2], fill=(64, 156, 214), width=rng.randint(1, 2))
            draw.ellipse([x2 - 3, y2 - 3, x2 + 3, y2 + 3], fill=(120, 220, 255))
        draw.rectangle([size * 0.12, size * 0.55, size * 0.88, size * 0.88],
                       outline=(90, 200, 240), width=2)

    elif topic == "politics":
        img = _gradient(size, (228, 224, 214), (186, 182, 174), rng)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, size, size * 0.18], fill=(178, 58, 48))
        draw.rectangle([0, size * 0.18, size, size * 0.28], fill=(240, 238, 232))
        draw.rectangle([0, size * 0.28, size, size * 0.42], fill=(46, 78, 138))
        draw.rectangle([size * 0.3, size * 0.62, size * 0.7, size], fill=(96, 84, 74))
        for i in range(9):
            x = 10 + i * (size // 9)
            y = size * 0.5 + rng.randrange(-8, 8)
            draw.ellipse([x, y, x + 16, y + 16], fill=(70, 66, 62))

    elif topic == "disaster":
        img = _gradient(size, (46, 26, 18), (196, 96, 42), rng)
        draw = ImageDraw.Draw(img)
        for _ in range(16):
            x, y = rng.randrange(size), rng.randrange(size)
            w, h = rng.randrange(30, 90), rng.randrange(24, 70)
            draw.ellipse([x, y, x + w, y + h], fill=(70 + rng.randrange(60), 46, 30))
        for _ in range(20):
            x, y = rng.randrange(size), rng.randrange(int(size * 0.6), size)
            draw.rectangle([x, y, x + rng.randrange(8, 26), y + rng.randrange(4, 14)],
                           fill=(48, 44, 40))
        img = img.filter(ImageFilter.GaussianBlur(0.8))

    elif topic == "sports":
        img = _gradient(size, (46, 128, 62), (28, 96, 48), rng)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, size, size * 0.3], fill=(26, 40, 58))
        for i in range(4):
            x = (i + 0.5) * size / 4
            draw.ellipse([x - 22, -18, x + 22, 26], fill=(240, 244, 200))
        draw.line([size * 0.5, size * 0.3, size * 0.5, size], fill=(236, 240, 232), width=3)
        draw.ellipse([size * 0.42, size * 0.62, size * 0.58, size * 0.62 + size * 0.16],
                     outline=(236, 240, 232), width=3)
        draw.ellipse([size * 0.66, size * 0.5, size * 0.72, size * 0.5 + 14],
                     fill=(240, 236, 220))

    else:  # pragma: no cover - defensive
        img = _gradient(size, (60, 60, 60), (140, 140, 140), rng)

    if variant % 3 == 0:
        img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


def apply_artefact(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    """Apply one synthetic editing artefact (visible to a CNN encoder)."""
    w, h = img.size
    if kind == "overlay":
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([0, int(h * 0.78), w, h], fill=(10, 10, 10, 205))
        draw.text((10, int(h * 0.84)), ARTEFACT_OVERLAY_TEXT, fill=(255, 236, 40))
        return img
    if kind == "splice":
        patch_w, patch_h = w // 3, h // 3
        x, y = rng.randrange(0, w - patch_w), rng.randrange(0, h - patch_h)
        noise = np.array(img.crop((x, y, x + patch_w, y + patch_h))).astype(np.int16)
        noise = np.clip(noise + rng.randint(-70, 70) + np.random.RandomState(
            rng.randrange(10_000)).randint(-40, 40, noise.shape), 0, 255).astype(np.uint8)
        img = img.copy()
        img.paste(Image.fromarray(noise), (x, y))
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + patch_w, y + patch_h], outline=(255, 255, 255), width=2)
        return img
    return img  # "compression" is applied at save time


def build_benchmark(seed: int = 42) -> List[dict]:
    """Build the full labelled sample list (deterministic for a given seed)."""
    rng = random.Random(seed)
    samples: List[dict] = []
    for topic in TOPICS:
        for text in CREDIBLE_CLAIMS[topic]:
            samples.extend(_expand(text, topic, label=0, rng=rng, matching=3))
        for text in MISLEADING_CLAIMS[topic]:
            samples.extend(_expand(text, topic, label=1, rng=rng, matching=2))
    rng.shuffle(samples)
    for idx, sample in enumerate(samples):
        sample["id"] = idx
    return samples


def _expand(text: str, topic: str, label: int, rng: random.Random,
            matching: int) -> List[dict]:
    out = []
    others = [t for t in TOPICS if t != topic]
    for k in range(matching):
        out.append({
            "text": text,
            "text_topic": topic,
            "image_topic": topic,
            "label": label,
            "consistency": 1,
            "artefact": _pick_artefact(rng, misleading=bool(label)),
            "variant": rng.randrange(60),
            "duplicate_index": k,
        })
    # One deliberately mismatched pair: an inconsistent pair is a misinformation signal.
    if rng.random() < 0.5:
        distractor = rng.choice(ADJACENT_TOPICS[topic])
    else:
        distractor = rng.choice(others)
    # Illustrative-image case: the text stands on its own, the picture is decoration.
    mismatch_label = 1
    if label == 0 and rng.random() < MISMATCH_ILLUSTRATIVE_RATE:
        mismatch_label = 0
    out.append({
        "text": text,
        "text_topic": topic,
        "image_topic": distractor,
        "label": mismatch_label,
        "consistency": 0,
        "artefact": _pick_artefact(rng, misleading=bool(label)),
        "variant": rng.randrange(60),
        "duplicate_index": matching,
    })
    for sample in out:
        sample["degraded"] = rng.random() < DEGRADED_RATE
    return out


def _pick_artefact(rng: random.Random, misleading: bool) -> str:
    p = 0.55 if misleading else 0.12
    if rng.random() > p:
        return "none"
    return rng.choice(["overlay", "splice", "compression"])


def materialise(samples: List[dict], out_dir, size: int = 224) -> List[str]:
    """Render every sample's image to disk and return the file paths."""
    import pathlib

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sample in samples:
        img = render_scene(sample["image_topic"], sample["variant"], size=size)
        rng = random.Random(hash((sample["id"], sample["artefact"])) & 0xFFFFFFFF)
        img = apply_artefact(img, sample["artefact"], rng)
        if sample.get("degraded"):
            # Poor-quality upload: aggressive downscale then upscale + blur.
            img = img.resize((56, 56), Image.BILINEAR).resize((size, size), Image.BILINEAR)
            img = img.filter(ImageFilter.GaussianBlur(1.1))
        path = out_dir / f"sample_{sample['id']:04d}.jpg"
        if sample["artefact"] == "compression":
            img.save(path, quality=32)
        else:
            img.save(path, quality=92)
        sample["image_path"] = str(path)
        paths.append(str(path))
    return paths


def describe() -> dict:
    credible = sum(len(v) for v in CREDIBLE_CLAIMS.values())
    misleading = sum(len(v) for v in MISLEADING_CLAIMS.values())
    return {
        "name": "Multimodal Proxy Benchmark (MPB-1)",
        "topics": TOPICS,
        "credible_claims": credible,
        "misleading_claims": misleading,
        "pairs": credible * 4 + misleading * 3,
        "image_source": "procedurally rendered topic scenes with optional editing artefacts",
        "degraded_image_rate": DEGRADED_RATE,
        "adjacent_topic_distractors": True,
        "mismatch_illustrative_rate": MISMATCH_ILLUSTRATIVE_RATE,
    }
