"""Explainability helpers: linguistic feature extraction and token attributions.

Token attributions use occlusion: each word is removed in turn and the fusion
classifier is re-run, so the reported impact is a measured change in the model's
misleading-probability, not a hand-written rule.
"""
from __future__ import annotations

import re
from typing import Callable, List

import numpy as np

SENSATIONAL_TERMS = {
    "breaking", "shocking", "exposed", "miracle", "secret", "secrets", "leaked",
    "banned", "hate", "viral", "forward", "urgent", "confirmed", "proof",
    "conspiracy", "hidden", "cure", "cures", "shameful", "outrage", "must",
    "share", "delete", "deleted", "insider", "bombshell", "terrifying",
}
URGENCY_TERMS = {"now", "tonight", "immediately", "urgent", "before", "hurry",
                 "deadline", "last", "today", "tomorrow", "fast"}
HEDGING_TERMS = {"allegedly", "reportedly", "may", "might", "could", "sources",
                 "unconfirmed", "appears", "likely"}
INSTITUTIONAL_TERMS = {"department", "ministry", "officials", "committee", "court",
                       "regulator", "commission", "board", "authorities", "hospital",
                       "university", "institute", "agency", "study", "researchers"}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "these", "those", "as", "has", "have", "had", "will", "would",
    "can", "could", "not", "no", "you", "your", "we", "our", "they", "their",
    "he", "she", "his", "her", "i", "my", "me", "us", "them", "than", "then",
    "there", "here", "which", "who", "whom", "so", "such", "into", "over",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d[\d.,]*%?")


def linguistic_features(text: str) -> dict:
    words = _WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    letters = [c for c in text if c.isalpha()]
    upper = [c for c in letters if c.isupper()]

    sensational = sorted({w for w in lower if w in SENSATIONAL_TERMS})
    urgency = sorted({w for w in lower if w in URGENCY_TERMS})
    hedging = sorted({w for w in lower if w in HEDGING_TERMS})
    institutional = sorted({w for w in lower if w in INSTITUTIONAL_TERMS})

    content = [w for w in lower if w not in STOPWORDS and len(w) > 2]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": max(1, len(sentences)),
        "avg_word_length": round(sum(len(w) for w in words) / max(1, len(words)), 2),
        "uppercase_ratio": round(len(upper) / max(1, len(letters)), 3),
        "exclamation_marks": text.count("!"),
        "question_marks": text.count("?"),
        "digit_ratio": round(sum(c.isdigit() for c in text) / max(1, len(text)), 3),
        "all_caps_words": sum(1 for w in words if len(w) > 2 and w.isupper()),
        "sensational_terms": sensational,
        "urgency_terms": urgency,
        "hedging_terms": hedging,
        "institutional_terms": institutional,
        "content_word_count": len(content),
        "lexical_diversity": round(len(set(content)) / max(1, len(content)), 3),
        "reads_like_forwarded_message": bool(
            re.search(r"\b(forward|share|send this|circulate)\b", text, re.I)
        ),
    }


def token_attributions(
    text: str,
    score_fn: Callable[[List[str]], np.ndarray],
    max_words: int = 26,
    top_negative: int = 6,
    top_positive: int = 4,
) -> dict:
    """Occlusion attributions.

    `score_fn` receives a list of texts and returns an array of
    P(misleading) for each. Removing a word and seeing the probability drop
    means the word supported the "misleading" decision.
    """
    tokens = text.split()
    if not tokens:
        return {"words": [], "baseline": None, "method": "occlusion"}

    baseline = float(score_fn([text])[0])

    keep_idx = [i for i in range(len(tokens))][:max_words]
    variants, kept = [], []
    for i in keep_idx:
        if tokens[i].strip().lower().strip(".,!?") in STOPWORDS:
            continue
        variant = " ".join(tokens[:i] + tokens[i + 1:]) or "empty"
        variants.append(variant)
        kept.append(i)
    if not variants:
        return {"words": [], "baseline": round(baseline, 4), "method": "occlusion"}

    masked = score_fn(variants)
    impacts = []
    for idx, masked_score, word in zip(kept, masked, [tokens[i] for i in kept]):
        clean = word.strip(".,!?\"'()")
        if len(clean) < 2:
            continue
        impact = baseline - float(masked_score)
        impacts.append({"token": clean, "impact": round(impact, 4)})

    negative = sorted([i for i in impacts if i["impact"] > 0.004],
                      key=lambda d: -d["impact"])[:top_negative]
    positive = sorted([i for i in impacts if i["impact"] < -0.004],
                      key=lambda d: d["impact"])[:top_positive]
    peak = max([abs(i["impact"]) for i in impacts] or [1e-6])
    for item in negative + positive:
        item["weight"] = round(abs(item["impact"]) / peak, 3)
    return {
        "words": negative + positive,
        "towards_misleading": negative,
        "towards_credible": positive,
        "baseline": round(baseline, 4),
        "words_tested": len(impacts),
        "method": "occlusion (leave-one-word-out) on the fusion classifier",
    }


def summarise(text_features: dict, image_flags: list, consistency: dict | None,
              misleading_prob: float, modality: str) -> str:
    parts = []
    if modality == "multimodal":
        parts.append(
            f"The fused representation places this post at {misleading_prob * 100:.1f}% "
            f"probability of being misleading."
        )
    else:
        parts.append(
            f"Only text was available, so the text branch of the network places this post "
            f"at {misleading_prob * 100:.1f}% probability of being misleading. This is a "
            f"single-modality result."
        )

    cue_bits = []
    if text_features["uppercase_ratio"] > 0.25 or text_features["all_caps_words"] >= 2:
        cue_bits.append("a high proportion of capitalised text")
    if text_features["exclamation_marks"] >= 2:
        cue_bits.append("repeated exclamation marks")
    if text_features["sensational_terms"]:
        cue_bits.append("sensational wording (" + ", ".join(
            text_features["sensational_terms"][:4]) + ")")
    if text_features["reads_like_forwarded_message"]:
        cue_bits.append("an explicit forward/share request")
    if text_features["institutional_terms"]:
        cue_bits.append("attribution to a named institution (" + ", ".join(
            text_features["institutional_terms"][:3]) + ")")
    if cue_bits:
        parts.append("Linguistically the post shows " + "; ".join(cue_bits) + ".")

    if consistency:
        parts.append(
            f"Cross-modal check: {consistency['label'].lower()} "
            f"(cosine {consistency['cosine_similarity']:+.3f}, "
            f"{consistency['score']:.0f}% probability of consistency)."
        )
    if image_flags:
        parts.append("Visual artefact indicators: " + ", ".join(
            f["name"].lower() for f in image_flags[:3]) + ".")
    return " ".join(parts)
