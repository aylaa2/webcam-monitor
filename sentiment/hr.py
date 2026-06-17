"""Multimodal HR interview analysis (face + voice + words), all non-LLM.

Produces, for the Interview Report window:
  - 6 SOFT SKILLS, each a transparent weighted blend of available modalities
  - HARD SKILLS actually mentioned (technical keywords)
  - per-skill EVIDENCE: the exact sentences the candidate said (shown via the
    clickable "more info" detail windows)
  - BACKGROUND highlights + KEY CONCEPTS mentioned
  - STAR-method coverage (Situation / Task / Action / Result)
  - voice/speech metrics, the emotion graph, and an HR recommendation
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from audio import classify, nlp
from audio.lexicon import FILLERS, NEGATIVE, POSITIVE, WordStats
from audio.nlp import STAR_CUES
from audio.prosody import ProsodyFeatures
from sentiment.report import _wrap, fmt_time

SOFT_ORDER = ["Communication", "Confidence", "Composure", "Enthusiasm",
              "Positivity", "Engagement"]
_BG = (30, 28, 26)
_FG = (238, 238, 238)
_DIM = (165, 165, 160)
_ACC = (120, 200, 140)
_HOT = (110, 150, 235)


def _clip(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _band(x: float, lo: float, hi: float, soft: float = 0.35) -> float:
    if x <= 0:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return _clip(1.0 - (lo - x) / (lo * soft))
    return _clip(1.0 - (x - hi) / (hi * soft))


def _blend(components: list[tuple[float, float]]) -> float:
    tot = sum(w for w, _ in components)
    return _clip(sum(w * v for w, v in components) / tot) if tot > 0 else 0.0


@dataclass
class Evidence:
    title: str
    score: float | None
    basis: list[str]
    quotes: list[tuple[float, str]]


@dataclass
class HRReport:
    soft: dict[str, float] = field(default_factory=dict)
    hard_skills: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    narrative: str = ""
    transcript: str = ""
    segments: list = field(default_factory=list)
    language: str = ""
    engine: str = ""
    emotion_description: str = ""
    emotion_moments: list = field(default_factory=list)
    background: list = field(default_factory=list)
    key_concepts: list = field(default_factory=list)
    star: nlp.Star = field(default_factory=nlp.Star)
    star_quotes: dict = field(default_factory=dict)
    topics: list = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    classified: bool = False
    voice_available: bool = False
    words_available: bool = False

    @property
    def overall(self) -> float:
        return sum(self.soft.values()) / len(self.soft) if self.soft else 0.0


def compute(face: dict, prosody: ProsodyFeatures, words: WordStats, duration_s: float,
            transcript: str = "", segments=None, language: str = "",
            engine: str = "") -> HRReport:
    segs = segments or []
    voice = prosody.available
    has_words = words.total_words > 0
    dur_min = max(1e-6, duration_s / 60.0)
    wpm = words.total_words / dur_min if has_words else 0.0

    eye = face.get("eye_contact_pct", 0.0) / 100.0
    engagement = face.get("mean_engagement", 0.0)
    composure_f = face.get("composure", 0.0)
    positivity_f = face.get("positivity", 0.0)
    expressiveness = face.get("expressiveness", 0.0)
    blink_calm = _clip(1.0 - (face.get("blink_rate", 0.0) - 30.0) / 30.0)

    energy = _clip(prosody.mean_energy / 0.22) if voice else 0.0
    pitch_cv = (prosody.std_f0 / prosody.mean_f0) if (voice and prosody.mean_f0 > 0) else 0.0
    pitch_steady = _clip(1.0 - pitch_cv / 0.45) if voice else 0.0
    pitch_var = _clip(prosody.f0_range / 120.0) if voice else 0.0
    fluency = _clip(1.0 - prosody.pause_ratio) if voice else 0.0
    voiced = prosody.voiced_ratio if voice else 0.0

    rate_score = _band(wpm, 110, 160) if has_words else 0.0
    filler_score = _clip(1.0 - words.filler_rate / 8.0) if has_words else 0.0
    vocab_score = _clip(words.vocab_richness / 0.55) if has_words else 0.0
    word_sent = (words.sentiment + 1.0) / 2.0 if has_words else 0.5
    pos_ratio = _clip(words.positive / max(1, words.total_words) * 20.0) if has_words else 0.0

    V = (lambda w, v: (w, v)) if voice else (lambda w, v: (0.0, 0.0))
    Wd = (lambda w, v: (w, v)) if has_words else (lambda w, v: (0.0, 0.0))

    soft = {
        "Communication": _blend([Wd(1.5, rate_score), Wd(1.5, filler_score),
                                 Wd(1.0, vocab_score), V(1.0, fluency), (0.8, expressiveness)]),
        "Confidence": _blend([(1.2, eye), V(1.0, fluency), V(1.0, energy),
                              V(0.8, pitch_steady), (0.8, blink_calm)]),
        "Composure": _blend([(1.2, composure_f), (1.0, blink_calm),
                             V(0.8, pitch_steady), Wd(0.8, filler_score)]),
        "Enthusiasm": _blend([V(1.0, energy), V(1.0, pitch_var), (1.0, positivity_f),
                             Wd(0.8, pos_ratio), (0.6, expressiveness)]),
        "Positivity": _blend([(1.2, positivity_f), Wd(1.0, word_sent)]),
        "Engagement": _blend([(1.2, engagement), (1.0, eye), V(0.8, voiced)]),
    }

    metrics = {
        "words": words.total_words, "wpm": wpm, "fillers": words.fillers,
        "filler_rate": words.filler_rate, "action_verbs": words.action_verbs,
        "vocab_richness": words.vocab_richness, "word_sentiment": words.sentiment,
        "mean_f0": prosody.mean_f0, "f0_range": prosody.f0_range,
        "pause_ratio": prosody.pause_ratio, "voiced_ratio": prosody.voiced_ratio,
        "energy": prosody.mean_energy,
    }

    # ---- per-skill evidence (quotes + basis) ----
    ev: dict[str, Evidence] = {}
    voice_line = (f"voice energy {energy * 100:.0f}, fluency {fluency * 100:.0f}, "
                  f"pitch steadiness {pitch_steady * 100:.0f}") if voice else "voice unavailable"
    ev["Communication"] = Evidence(
        "Communication", soft["Communication"],
        [f"pace: {wpm:.0f} wpm (ideal 110-160)" if has_words else "speech unavailable",
         f"filler words: {words.fillers} ({words.filler_rate:.1f} per 100)" if has_words else "",
         f"vocabulary richness: {words.vocab_richness:.2f}" if has_words else "",
         f"pauses: {prosody.pause_ratio * 100:.0f}%" if voice else ""],
        nlp.find_quotes(segs, FILLERS))
    ev["Confidence"] = Evidence(
        "Confidence", soft["Confidence"],
        [f"eye contact: {eye * 100:.0f}%", voice_line,
         f"calm blink rate score: {blink_calm * 100:.0f}"],
        nlp.find_quotes(segs, POSITIVE))
    ev["Composure"] = Evidence(
        "Composure", soft["Composure"],
        [f"facial composure: {composure_f * 100:.0f}", f"blink calm: {blink_calm * 100:.0f}",
         f"pitch steadiness: {pitch_steady * 100:.0f}" if voice else "",
         f"filler words: {words.fillers}" if has_words else ""],
        nlp.find_quotes(segs, FILLERS))
    ev["Enthusiasm"] = Evidence(
        "Enthusiasm", soft["Enthusiasm"],
        [f"vocal energy: {energy * 100:.0f}" if voice else "",
         f"pitch range: {prosody.f0_range:.0f} Hz" if voice else "",
         f"facial positivity: {positivity_f * 100:.0f}",
         f"positive words: {words.positive}" if has_words else ""],
        nlp.find_quotes(segs, POSITIVE))
    ev["Positivity"] = Evidence(
        "Positivity", soft["Positivity"],
        [f"facial positivity: {positivity_f * 100:.0f}",
         f"word sentiment: {words.sentiment:+.2f}" if has_words else "",
         f"positive/negative words: {words.positive}/{words.negative}" if has_words else ""],
        nlp.find_quotes(segs, POSITIVE | NEGATIVE))
    ev["Engagement"] = Evidence(
        "Engagement", soft["Engagement"],
        [f"engagement: {engagement * 100:.0f}", f"eye contact: {eye * 100:.0f}%",
         f"voiced ratio: {voiced * 100:.0f}%" if voice else ""],
        segs[:3] and [(s.start, s.text) for s in segs[:3]] or [])
    for cat, kws in words.hard_skills.items():
        ev[f"hard:{cat}"] = Evidence(cat, None, [f"mentioned: {', '.join(kws)}"],
                                     nlp.find_quotes(segs, set(kws)))
    for e in ev.values():
        e.basis = [b for b in e.basis if b]

    # ---- semantic classification (multilingual) + lexicon fallback ----
    star = nlp.detect_star(transcript)
    star_quotes: dict[str, list] = {}
    bg = nlp.background(segs)
    topics: list = []
    cls = classify.classify(segs)
    if cls:
        for comp in ("Situation", "Task", "Action", "Result"):
            star.present[comp] = star.present.get(comp, False) or cls.star_present.get(comp, False)
        star_quotes = {k: list(v) for k, v in cls.star_quotes.items()}
        bg = cls.background + bg
        for disp, kind, quotes in cls.topics:
            topics.append((disp, kind))
            ev[f"topic:{disp}"] = Evidence(disp, None,
                                          [f"{kind.lower()} topic - {len(quotes)} mention(s)"],
                                          quotes[:5])

    for comp in ("Situation", "Task", "Action", "Result"):
        q = list(star_quotes.get(comp, []))
        if len(q) < 2:
            q += nlp.find_phrase_quotes(segs, STAR_CUES.get(comp, set()), 3)
        seen, dq = set(), []
        for t, x in q:
            if x not in seen:
                seen.add(x)
                dq.append((t, x))
        star_quotes[comp] = dq[:4]

    seen, bg2 = set(), []
    for t, x in bg:
        if x not in seen:
            seen.add(x)
            bg2.append((t, x))

    report = HRReport(
        soft=soft, hard_skills=words.hard_skills, metrics=metrics, transcript=transcript,
        segments=segs, language=language, engine=engine,
        background=bg2[:6], key_concepts=nlp.key_concepts(transcript, words.hard_skills),
        star=star, star_quotes=star_quotes, topics=topics, classified=cls is not None,
        evidence=ev, voice_available=voice, words_available=has_words)
    report.narrative = _narrative(report, face, duration_s)
    return report


def _narrative(r: HRReport, face: dict, duration_s: float) -> str:
    parts: list[str] = []
    overall = r.overall * 100
    verdict = ("a strong candidate" if overall >= 70 else
               "a solid candidate" if overall >= 55 else
               "a developing candidate with clear growth areas")
    parts.append(f"Across a {fmt_time(duration_s)} interview, this reads as {verdict} "
                 f"(overall {overall:.0f}/100).")
    ranked = sorted(r.soft.items(), key=lambda kv: -kv[1])
    strengths = [k for k, v in ranked[:2] if v >= 0.55]
    if strengths:
        parts.append(f"Strongest: {', '.join(s.lower() for s in strengths)}.")
    weak = [k for k, v in ranked[-2:] if v < 0.5]
    if weak:
        parts.append(f"Probe further: {', '.join(w.lower() for w in weak)}.")
    flags: list[str] = []
    if r.words_available and r.metrics["filler_rate"] > 6:
        flags.append(f"frequent fillers ({r.metrics['filler_rate']:.0f}/100)")
    if r.words_available and r.metrics["wpm"] > 175:
        flags.append("fast pace")
    elif r.words_available and 0 < r.metrics["wpm"] < 95:
        flags.append("slow/hesitant pace")
    if face.get("eye_contact_pct", 0) < 40:
        flags.append("limited eye contact")
    if r.voice_available and r.metrics["f0_range"] < 25:
        flags.append("monotone delivery")
    if flags:
        parts.append("Watch-outs: " + ", ".join(flags) + ".")
    star_present = [k for k, v in r.star.present.items() if v]
    if r.words_available:
        parts.append(f"STAR coverage: {len(star_present)}/4 "
                     f"({', '.join(star_present) if star_present else 'none detected'}).")
    if r.hard_skills:
        parts.append("Technical areas: " + ", ".join(r.hard_skills) + ".")
    parts.append(f"Dominant facial affect: '{face.get('dominant_emotion', 'neutral')}'.")
    if not r.voice_available:
        parts.append("(Voice unavailable — scores are face-only.)")
    return " ".join(parts)


# ----------------------------------------------------------------- rendering
def render_image(r: HRReport, face: dict, graph_path=None):
    """Returns (BGR image, buttons) where buttons = [(x1,y1,x2,y2,key), ...].
    Text is drawn via a Unicode font layer so Romanian diacritics render."""
    import cv2
    from pathlib import Path

    from vision.textrender import TextLayer

    Lw, Rw, gap, pad, title_h = 700, 720, 16, 24, 56
    W = pad + Lw + gap + Rw + pad

    def newcol(w):
        return np.full((2800, w, 3), _BG, np.uint8)

    def bar(c, x, y, w, frac):
        cv2.rectangle(c, (x, y), (x + w, y + 13), (70, 68, 64), 1)
        col = _ACC if frac >= 0.55 else (90, 160, 230) if frac >= 0.4 else (90, 90, 230)
        cv2.rectangle(c, (x, y), (x + int(w * frac), y + 13), col, -1)

    # ---------- LEFT ----------
    L = newcol(Lw)
    t = TextLayer()
    Lb: list[tuple] = []
    y = 22
    t.put(f"Overall  {r.overall * 100:.0f}/100", 4, y, _ACC, 0.7, True)
    mods = "face" + (" + voice" if r.voice_available else "") + \
           (" + words" if r.words_available else "")
    lang = f"  ({r.language})" if r.language else ""
    t.put(f"signals: {mods}{lang}", 250, y, _DIM, 0.45)
    y += 14
    cv2.line(L, (4, y), (Lw - 8, y), (70, 68, 64), 1)
    y += 30
    t.put("SOFT SKILLS   (click a skill for what was said)", 4, y, _ACC, 0.5)
    y += 24
    for k in SOFT_ORDER:
        t.put(k, 4, y - 6, _DIM, 0.45)
        t.put("more >", Lw - 74, y - 6, _HOT, 0.42)
        bar(L, 4, y, 520, r.soft.get(k, 0.0))
        t.put(f"{r.soft.get(k, 0.0) * 100:.0f}", 532, y + 12, _FG, 0.5)
        Lb.append((0, y - 20, Lw, y + 18, f"soft:{k}"))
        y += 38
    y += 8
    t.put("VOICE & SPEECH", 4, y, _ACC, 0.55, True)
    y += 26
    m = r.metrics
    if r.words_available:
        for line in (f"words: {int(m['words'])}    pace: {m['wpm']:.0f} wpm",
                     f"filler words: {int(m['fillers'])} ({m['filler_rate']:.1f}/100)    "
                     f"impact verbs: {int(m['action_verbs'])}",
                     f"vocab richness: {m['vocab_richness']:.2f}    "
                     f"sentiment: {m['word_sentiment']:+.2f}"):
            t.put(line, 8, y, _FG, 0.5)
            y += 22
    if r.voice_available:
        t.put(f"pitch: {m['mean_f0']:.0f} Hz (range {m['f0_range']:.0f})   "
              f"pauses: {m['pause_ratio'] * 100:.0f}%   voiced: {m['voiced_ratio'] * 100:.0f}%",
              8, y, _FG, 0.5)
        y += 22
    if not r.voice_available and not r.words_available:
        t.put("voice/words unavailable (no mic or model)", 8, y, _DIM, 0.5)
        y += 22
    y += 8
    t.put("STAR METHOD", 4, y, _ACC, 0.55, True)
    t.put("more >", Lw - 74, y, _HOT, 0.42)
    Lb.append((0, y - 18, Lw, y + 10, "section:star"))
    y += 26
    sx = 8
    for comp in ("Situation", "Task", "Action", "Result"):
        ok = r.star.present.get(comp, False)
        t.put(f"{'[x]' if ok else '[ ]'} {comp}", sx, y, _ACC if ok else (90, 90, 230), 0.5)
        sx += 170
    y += 6
    tag = "  (semantic + cues)" if r.classified else ""
    t.put(f"coverage {sum(r.star.present.values())}/4{tag}", 8, y + 14, _DIM, 0.45)
    y += 32
    t.put("RECOMMENDATION", 4, y, _ACC, 0.55, True)
    y += 24
    for ln in _wrap(r.narrative, 80):
        t.put(ln, 8, y, _FG, 0.46)
        y += 20
    Lh = y + pad
    L = t.render(L)[:Lh]

    # ---------- RIGHT ----------
    R = newcol(Rw)
    t = TextLayer()
    Rb: list[tuple] = []
    y = 22
    t.put("FACIAL AFFECT", 4, y, _ACC, 0.55, True)
    y += 24
    for line in (f"dominant emotion: {face.get('dominant_emotion', '-')}",
                 f"engagement {face.get('mean_engagement', 0) * 100:.0f}   "
                 f"eye contact {face.get('eye_contact_pct', 0):.0f}%   "
                 f"composure {face.get('composure', 0) * 100:.0f}"):
        t.put(line, 8, y, _FG, 0.5)
        y += 22
    y += 8
    if r.emotion_description:
        t.put("HOW THEY FELT", 4, y, _ACC, 0.55, True)
        y += 24
        for ln in _wrap(r.emotion_description, 84):
            t.put(ln, 8, y, _FG, 0.45)
            y += 19
        y += 8
    if r.emotion_moments:
        t.put("PEAK EMOTIONAL MOMENTS", 4, y, _ACC, 0.55, True)
        y += 24
        for tm, emo, v in r.emotion_moments:
            t.put(f"{fmt_time(tm):>6}   {emo:<10} {v * 100:3.0f}%", 8, y, _HOT, 0.5)
            y += 20
        y += 8
    if r.topics:
        t.put("TOPICS DISCUSSED   (click for what was said)", 4, y, _ACC, 0.5)
        y += 24
        for disp, kind in r.topics:
            t.put(f"[{kind}] {disp}", 8, y, _HOT, 0.5)
            t.put("more >", Rw - 74, y, _HOT, 0.42)
            Rb.append((0, y - 16, Rw, y + 8, f"topic:{disp}"))
            y += 22
        y += 8
    t.put("HARD SKILLS (keywords)   (click for quotes)", 4, y, _ACC, 0.5)
    y += 24
    if r.hard_skills:
        for cat, kws in r.hard_skills.items():
            t.put(cat, 8, y, _HOT, 0.5)
            t.put(", ".join(kws)[:55], 210, y, _FG, 0.46)
            t.put("more >", Rw - 74, y, _HOT, 0.42)
            Rb.append((0, y - 16, Rw, y + 8, f"hard:{cat}"))
            y += 22
    else:
        t.put("none detected", 8, y, _DIM, 0.5)
        y += 22
    y += 8
    if r.key_concepts:
        t.put("KEY CONCEPTS", 4, y, _ACC, 0.55, True)
        y += 24
        concept_str = "  ".join(f"{w}({c})" for w, c in r.key_concepts)
        for ln in _wrap(concept_str, 86):
            t.put(ln, 8, y, _FG, 0.46)
            y += 19
        y += 8
    for label, key, bx in (("Background >", "section:background", 4),
                           ("Full transcript >", "section:transcript", 250)):
        cv2.rectangle(R, (bx, y - 15), (bx + 200, y + 11), (60, 90, 60), -1)
        t.put(label, bx + 10, y + 4, _FG, 0.5)
        Rb.append((bx, y - 15, bx + 200, y + 11, key))
    y += 30
    if graph_path and Path(graph_path).exists():
        g = cv2.imread(str(graph_path))
        if g is not None:
            gh = int(Rw * g.shape[0] / g.shape[1])
            R[y:y + gh, 0:Rw] = cv2.resize(g, (Rw, gh))
            y += gh
    Rh = y + pad
    R = t.render(R)[:Rh]

    H = title_h + max(Lh, Rh) + pad
    canvas = np.full((H, W, 3), _BG, np.uint8)
    cv2.line(canvas, (pad, title_h - 4), (W - pad, title_h - 4), (70, 68, 64), 1)
    canvas[title_h:title_h + Lh, pad:pad + Lw] = L
    canvas[title_h:title_h + Rh, pad + Lw + gap:pad + Lw + gap + Rw] = R
    tt = TextLayer()
    tt.put("INTERVIEW REPORT  -  HR ANALYSIS", pad, 40, _ACC, 0.9, True)
    canvas = tt.render(canvas)

    buttons = [(pad + x1, title_h + y1, pad + x2, title_h + y2, k)
               for (x1, y1, x2, y2, k) in Lb]
    buttons += [(pad + Lw + gap + x1, title_h + y1, pad + Lw + gap + x2, title_h + y2, k)
                for (x1, y1, x2, y2, k) in Rb]
    return canvas, buttons


def render_detail(r: HRReport, key: str, width: int = 940) -> np.ndarray:
    import cv2

    from vision.textrender import TextLayer

    canvas = np.full((3200, width, 3), _BG, np.uint8)
    t = TextLayer()
    y = [36]

    def title(s):
        t.put(s, 24, y[0], _ACC, 0.8, True)
        y[0] += 14
        cv2.line(canvas, (24, y[0]), (width - 24, y[0]), (70, 68, 64), 1)
        y[0] += 26

    def quotes(qs):
        if not qs:
            t.put("(no specific lines captured)", 28, y[0], _DIM, 0.5)
            y[0] += 22
            return
        for tm, text in qs:
            t.put(f"[{fmt_time(tm)}]", 28, y[0], _HOT, 0.5)
            for ln in _wrap(text, 92):
                t.put(ln, 96, y[0], _FG, 0.48)
                y[0] += 19
            y[0] += 6

    if key.startswith(("soft:", "hard:", "topic:")):
        ev = r.evidence.get(key[5:] if key.startswith("soft:") else key)
        if ev is None:
            title("No detail")
        else:
            head = ev.title + (f"   {ev.score * 100:.0f}/100" if ev.score is not None else "")
            title(head)
            if ev.basis:
                t.put("Based on:", 24, y[0], _ACC, 0.55, True)
                y[0] += 24
                for b in ev.basis:
                    t.put(f"- {b}", 28, y[0], _FG, 0.5)
                    y[0] += 21
                y[0] += 10
            t.put("What was said:", 24, y[0], _ACC, 0.55, True)
            y[0] += 26
            quotes(ev.quotes)
    elif key == "section:star":
        title("STAR method breakdown")
        for comp in ("Situation", "Task", "Action", "Result"):
            ok = r.star.present.get(comp, False)
            t.put(f"{'[x]' if ok else '[ ]'} {comp}", 24, y[0],
                  _ACC if ok else (90, 90, 230), 0.6, True)
            y[0] += 26
            quotes(r.star_quotes.get(comp, []))
            y[0] += 6
    elif key == "section:background":
        title("Background & experience mentioned")
        quotes(r.background)
    elif key == "section:transcript":
        title(f"Full transcript  ({r.engine}{', ' + r.language if r.language else ''})")
        if r.transcript:
            for ln in _wrap(r.transcript, 104):
                t.put(ln, 24, y[0], _FG, 0.48)
                y[0] += 19
        else:
            t.put("(no transcript - voice was unavailable)", 24, y[0], _DIM, 0.5)
            y[0] += 20
    else:
        title("Detail")

    canvas = t.render(canvas)
    return canvas[:y[0] + 24]
