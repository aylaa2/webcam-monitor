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

import os
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
    ("hard:web", "Hard", "Web / Frontend",
     ["web development, frontend, backend, react and APIs"]),
    ("hard:cloud", "Hard", "Cloud / DevOps",
     ["cloud computing, deployment, docker, kubernetes and devops"]),
    ("hard:databases", "Hard", "Databases",
     ["databases, SQL queries and data storage"]),
    ("hard:design", "Hard", "Design",
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


def _classify_embed(segments) -> Classification | None:
    if not _load():
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


def keyphrases(text: str, top: int = 10, diversity: float = 0.65
               ) -> list[tuple[str, int]] | None:
    """Semantic key-concept extraction (KeyBERT-style + MMR).

    Candidates are uni/bi/tri-grams of content words; they're ranked by cosine
    similarity of their embedding to the whole-document embedding, then selected
    with Maximal Marginal Relevance so the list is precise AND non-redundant.
    Domain terms (known skill keywords) and multi-word phrases get a small boost.
    None if the embedding model is unavailable (callers fall back to frequency)."""
    if not _load():
        return None
    from collections import Counter

    from audio.lexicon import HARD_SKILLS, tokenize
    from audio.nlp import STOPWORDS

    toks = tokenize(text)
    content = [w for w in toks if len(w) > 2 and w not in STOPWORDS]
    cands: set[str] = set(content)
    for a, b in zip(toks, toks[1:]):
        if len(a) > 2 and len(b) > 2 and a not in STOPWORDS and b not in STOPWORDS:
            cands.add(f"{a} {b}")
    for a, b, c in zip(toks, toks[1:], toks[2:]):           # trigrams
        if all(len(w) > 2 for w in (a, c)) and a not in STOPWORDS and c not in STOPWORDS:
            cands.add(f"{a} {b} {c}")
    cands = [c for c in cands if not c.isdigit()]
    if not cands:
        return []

    counts = Counter(content)
    domain = {kw for kws in HARD_SKILLS.values() for kw in kws}
    doc = _embed([text])[0]
    cand_emb = _embed(cands)
    rel = cand_emb @ doc
    boost = np.array([(0.05 if any(w in domain for w in c.split()) else 0.0)
                      + (0.02 if " " in c else 0.0) for c in cands], dtype=np.float32)
    score = rel + boost

    selected: list[int] = []
    remaining = list(range(len(cands)))
    while remaining and len(selected) < top:
        if not selected:
            i = max(remaining, key=lambda k: score[k])
        else:
            def mmr(k):
                sim = max(float(cand_emb[k] @ cand_emb[j]) for j in selected)
                return diversity * score[k] - (1 - diversity) * sim
            i = max(remaining, key=mmr)
        remaining.remove(i)
        p = cands[i]
        if any(p in cands[j] or cands[j] in p for j in selected):  # sub/superstring dedup
            continue
        selected.append(i)

    out = []
    for i in selected:
        p = cands[i]
        cnt = max(1, text.lower().count(p)) if " " in p else counts.get(p, 1)
        out.append((p, int(cnt)))
    return out


# -------------------- cross-encoder reranker path (opt-in via CLASSIFY_RERANK=1) ------
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
STAR_MARGIN = 0.5
TOPIC_MARGIN = 0.5

_rerank = None
_rerank_unavail = False


def _load_rerank() -> bool:
    global _rerank, _rerank_unavail
    if _rerank is not None:
        return True
    if _rerank_unavail:
        return False
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _rerank = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=str(CACHE_DIR))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[classify] cross-encoder unavailable: {exc}")
        _rerank_unavail = True
        return False


def _classify_rerank(segments) -> Classification:
    star_docs = [" ".join(p) for _, p in STAR_PROTOS]
    topic_docs = [" ".join(p) for *_, p in TOPIC_PROTOS]
    all_docs = star_docs + topic_docs
    n_star = len(star_docs)
    star_idx_other = [i for i, (k, _) in enumerate(STAR_PROTOS) if k == "_other"][0]
    topic_idx_other = [i for i, (k, *_2) in enumerate(TOPIC_PROTOS) if k == "other"][0]

    star_present = {c: False for c in ("Situation", "Task", "Action", "Result")}
    star_quotes: dict[str, list] = defaultdict(list)
    background: list = []
    topic_q: dict[str, list] = defaultdict(list)
    topic_kind: dict[str, str] = {}

    for s in segments:
        scores = np.array(list(_rerank.rerank(s.text, all_docs)), dtype=np.float32)
        ss, st = scores[:n_star], scores[n_star:]
        quote = (s.start, s.text)

        best_j, best_v = -1, -1e9
        for i, (k, _) in enumerate(STAR_PROTOS):
            if k != "_other" and ss[i] > best_v:
                best_v, best_j = ss[i], i
        if best_v - ss[star_idx_other] > STAR_MARGIN:
            comp = STAR_PROTOS[best_j][0]
            star_present[comp] = True
            star_quotes[comp].append(quote)

        best_k, best_tv = -1, -1e9
        for i, (k, *_2) in enumerate(TOPIC_PROTOS):
            if k != "other" and st[i] > best_tv:
                best_tv, best_k = st[i], i
        if best_tv - st[topic_idx_other] > TOPIC_MARGIN:
            key, kind, disp, _ = TOPIC_PROTOS[best_k]
            if kind == "Background":
                background.append(quote)
            else:
                topic_q[disp].append(quote)
                topic_kind[disp] = kind

    topics = sorted(((d, topic_kind[d], q) for d, q in topic_q.items()),
                    key=lambda t: -len(t[2]))
    return Classification(star_present=star_present, star_quotes=dict(star_quotes),
                          background=background, topics=topics)


def classify(segments) -> Classification | None:
    """Bi-encoder embeddings by default (empirically best on this prototype-based
    multi-class task). The cross-encoder reranker is available as an opt-in
    experiment via CLASSIFY_RERANK=1 (it needs prototype/threshold tuning to win
    here, so it is off by default)."""
    if not segments:
        return None
    if os.environ.get("CLASSIFY_RERANK") == "1" and _load_rerank():
        try:
            return _classify_rerank(segments)
        except Exception as exc:  # noqa: BLE001
            print(f"[classify] reranker failed ({exc}); falling back to embeddings.")
    return _classify_embed(segments)
