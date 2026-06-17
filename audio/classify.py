"""Multilingual semantic classifier for transcript segments.

Instead of keyword lists, each thing the candidate says is embedded with a
multilingual sentence-embedding model (fastembed / paraphrase-multilingual-mpnet)
and matched by cosine similarity to short prototype descriptions of each
category. This determines — for Romanian OR English, by meaning not keywords —
whether a segment is a STAR component, background, or a soft/hard-skill topic.

Two independent passes are used because a single sentence can be BOTH a STAR
"Action" and a "Programming" topic (e.g. "I implemented caching with Redis"):
  - STAR pass   : Situation / Task / Action / Result vs "not a STAR element"
  - TOPIC pass  : background / soft skill / hard skill vs "other"

It's a transformer *encoder* (embeddings), not a generative LLM, and it degrades
gracefully: if fastembed/the model is missing, callers fall back to the lexicon
rules in audio/nlp.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "embed"
STAR_THRESHOLD = 0.30
TOPIC_THRESHOLD = 0.36

STAR_PROTOS: list[tuple[str, list[str]]] = [
    ("Situation", ["describing the situation and context of a past project or challenge",
                   "at my previous job there was a problem or difficult situation we faced"]),
    ("Task", ["explaining the task, goal or responsibility I was given",
              "what I needed to accomplish, the objective I was responsible for"]),
    ("Action", ["the concrete actions and steps I personally took to solve the problem",
                "I decided to, my approach was, the steps I followed were"]),
    ("Result", ["the result, outcome and impact, with numbers or improvements",
                "as a result we increased, reduced or improved something measurable"]),
    ("_other", ["just naming a technology, a greeting, or a general statement that does "
                "not narrate a situation, task, action or result"]),
]

# (key, kind, display, prototype phrases)
TOPIC_PROTOS: list[tuple[str, str, str, list[str]]] = [
    ("background", "Background", "Background / experience",
     ["my background, education, studies and years of experience",
      "my career history, the companies and roles I worked in before"]),
    ("soft:communication", "Soft", "Communication",
     ["communicating, presenting and explaining ideas clearly to people"]),
    ("soft:teamwork", "Soft", "Teamwork",
     ["working in a team, collaborating and helping colleagues"]),
    ("soft:leadership", "Soft", "Leadership",
     ["leading, managing or mentoring people and taking ownership"]),
    ("soft:problem_solving", "Soft", "Problem-solving",
     ["analytical thinking and solving difficult or complex problems"]),
    ("soft:adaptability", "Soft", "Adaptability",
     ["adapting to change, learning quickly and being flexible"]),
    ("hard:programming", "Hard", "Programming",
     ["programming languages and writing software code, python java c++"]),
    ("hard:data_ml", "Hard", "Data / ML",
     ["machine learning, data analysis, statistics and artificial intelligence"]),
    ("hard:web", "Hard", "Web development",
     ["web development, frontend, backend, react and APIs"]),
    ("hard:cloud", "Hard", "Cloud / DevOps",
     ["cloud computing, deployment, docker, kubernetes and devops"]),
    ("hard:databases", "Hard", "Databases",
     ["databases, SQL queries and data storage"]),
    ("hard:design", "Hard", "Design / UX",
     ["design, user interface, user experience and prototyping"]),
    ("other", "Other", "Other", ["greetings, thanks, small talk and unrelated chit chat"]),
]

_model = None
_star_lbl: list[str] = []
_star_mat: np.ndarray | None = None
_topic_lbl: list[tuple[str, str, str]] = []
_topic_mat: np.ndarray | None = None


@dataclass
class Classification:
    star_present: dict[str, bool] = field(default_factory=dict)
    star_quotes: dict[str, list[tuple[float, str]]] = field(default_factory=dict)
    background: list[tuple[float, str]] = field(default_factory=list)
    topics: list[tuple[str, str, list[tuple[float, str]]]] = field(default_factory=list)


def _embed_protos(phrase_lists: list[list[str]]) -> np.ndarray:
    mat = []
    for phrases in phrase_lists:
        e = np.array(list(_model.embed(phrases)), dtype=np.float32)
        v = e.mean(axis=0)
        mat.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(mat, dtype=np.float32)


def _load() -> bool:
    global _model, _star_lbl, _star_mat, _topic_lbl, _topic_mat
    if _model is not None:
        return True
    try:
        from fastembed import TextEmbedding
    except Exception:  # noqa: BLE001
        return False
    try:
        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
        _star_lbl = [c for c, _ in STAR_PROTOS]
        _star_mat = _embed_protos([p for _, p in STAR_PROTOS])
        _topic_lbl = [(k, kind, d) for k, kind, d, _ in TOPIC_PROTOS]
        _topic_mat = _embed_protos([p for *_, p in TOPIC_PROTOS])
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[classify] embedding model unavailable: {exc}")
        return False


def available() -> bool:
    return _load()


def _embed(texts: list[str]) -> np.ndarray:
    e = np.array(list(_model.embed(texts)), dtype=np.float32)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def classify(segments) -> Classification | None:
    if not segments or not _load():
        return None
    embs = _embed([s.text for s in segments])
    star_sims = embs @ _star_mat.T
    topic_sims = embs @ _topic_mat.T

    star_present = {c: False for c in ("Situation", "Task", "Action", "Result")}
    star_quotes: dict[str, list] = defaultdict(list)
    background: list = []
    topic_q: dict[str, list] = defaultdict(list)
    topic_kind: dict[str, str] = {}

    for i, s in enumerate(segments):
        quote = (s.start, s.text)
        js = int(np.argmax(star_sims[i]))
        comp = _star_lbl[js]
        if comp != "_other" and float(star_sims[i][js]) >= STAR_THRESHOLD:
            star_present[comp] = True
            star_quotes[comp].append(quote)
        jt = int(np.argmax(topic_sims[i]))
        key, kind, disp = _topic_lbl[jt]
        if float(topic_sims[i][jt]) >= TOPIC_THRESHOLD:
            if kind == "Background":
                background.append(quote)
            elif kind in ("Soft", "Hard"):
                topic_q[disp].append(quote)
                topic_kind[disp] = kind

    topics = sorted(((d, topic_kind[d], q) for d, q in topic_q.items()),
                    key=lambda t: -len(t[2]))
    return Classification(star_present=star_present, star_quotes=dict(star_quotes),
                          background=background, topics=topics)
