"""End-of-session interview analysis.

Accumulates per-frame readings, then produces:
  - a behavioral/affective PROFILE of the person (engagement, attentiveness,
    positivity, expressiveness, composure, dominant emotion, variability)
  - a plain-language DESCRIPTION of how they felt during the recording
    (template-based — generated from the statistics, no LLM)
  - the MOST EMOTIONAL MOMENTS with timestamps
  - an annotated emotion-over-time GRAPH (PNG) + a text report

Note: we deliberately profile *affect and behavior*, not demographics — age/sex
estimation from a webcam is unreliable and ethically loaded, so it's left out.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .blendshape_emotion import EMOTIONS  # noqa: E402
from .fusion import Engagement  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
_NON_NEUTRAL = [e for e in EMOTIONS if e != "neutral"]


def fmt_time(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


@dataclass
class Recorder:
    t0: float = field(default_factory=time.monotonic)
    t: list[float] = field(default_factory=list)
    emotions: list[dict[str, float]] = field(default_factory=list)
    top_emotion: list[str] = field(default_factory=list)
    engagement: list[Engagement] = field(default_factory=list)
    valence: list[float] = field(default_factory=list)
    arousal: list[float] = field(default_factory=list)
    gaze_x: list[float] = field(default_factory=list)
    gaze_y: list[float] = field(default_factory=list)
    yaw: list[float] = field(default_factory=list)
    pitch: list[float] = field(default_factory=list)
    blink_total: int = 0

    def add(self, emo, eng: Engagement, valence: float, arousal: float,
            gaze_x: float = 0.0, gaze_y: float = 0.0, yaw: float = 0.0,
            pitch: float = 0.0) -> None:
        self.t.append(time.monotonic() - self.t0)
        self.emotions.append(emo)
        self.top_emotion.append(max(emo, key=emo.get))
        self.engagement.append(eng)
        self.valence.append(valence)
        self.arousal.append(arousal)
        self.gaze_x.append(gaze_x)
        self.gaze_y.append(gaze_y)
        self.yaw.append(yaw)
        self.pitch.append(pitch)

    # ----------------------------------------------------------------- stats
    def _avg(self, attr: str) -> float:
        vals = [getattr(e, attr) for e in self.engagement]
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict:
        if not self.t:
            return {}
        n = len(self.t)
        # expressiveness: how far from neutral, on average.
        expressiveness = sum(1.0 - f.get("neutral", 0.0) for f in self.emotions) / n
        # variability: how often the dominant emotion switched.
        switches = sum(1 for a, b in zip(self.top_emotion, self.top_emotion[1:]) if a != b)
        variability = switches / max(1, n - 1)
        dominant = Counter(self.top_emotion).most_common(1)[0][0]
        dur = self.t[-1]
        # eye contact = % of frames the subject was clearly looking at the camera.
        looking = sum(1 for e in self.engagement if e.eye_contact >= 0.5) / n
        return {
            "duration_s": dur,
            "frames": n,
            "dominant_emotion": dominant,
            "mean_engagement": self._avg("overall"),
            "attentiveness": self._avg("facing"),
            "eye_contact_pct": 100.0 * looking,
            "positivity": self._avg("positivity"),
            "composure": self._avg("composure"),
            "mean_valence": sum(self.valence) / n,
            "expressiveness": expressiveness,
            "variability": variability,
            "blinks": self.blink_total,
            "blink_rate": self.blink_total / (dur / 60.0) if dur else 0.0,
        }

    def emotional_moments(self, top_k: int = 4, min_gap_s: float = 2.5,
                          min_intensity: float = 0.30) -> list[tuple[float, str, float]]:
        """Peak non-neutral expression moments, spaced out in time."""
        if not self.t:
            return []
        intens = []
        for f in self.emotions:
            e = max(_NON_NEUTRAL, key=lambda k: f.get(k, 0.0))
            intens.append((f.get(e, 0.0), e))
        order = sorted(range(len(intens)), key=lambda i: -intens[i][0])
        chosen: list[int] = []
        for i in order:
            if intens[i][0] < min_intensity:
                break
            if all(abs(self.t[i] - self.t[j]) >= min_gap_s for j in chosen):
                chosen.append(i)
            if len(chosen) >= top_k:
                break
        chosen.sort(key=lambda i: self.t[i])
        return [(self.t[i], intens[i][1], intens[i][0]) for i in chosen]

    def emotion_peaks(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for emo in _NON_NEUTRAL:
            series = [f.get(emo, 0.0) for f in self.emotions]
            if series and max(series) > 0:
                i = int(np.argmax(series))
                out[emo] = (self.t[i], series[i])
        return out

    # ----------------------------------------------------------------- text
    def describe(self) -> str:
        s = self.summary()
        if not s:
            return "No data captured."
        parts: list[str] = []

        val = s["mean_valence"]
        mood = ("positive and warm" if val > 0.12 else
                "tense or downbeat" if val < -0.12 else "fairly neutral and even")
        parts.append(
            f"Over {fmt_time(s['duration_s'])} of recording, the subject came across as "
            f"{mood}, with \"{s['dominant_emotion']}\" their most frequent expression."
        )

        eng = s["mean_engagement"]
        eng_word = ("highly engaged" if eng > 0.7 else
                    "moderately engaged" if eng > 0.5 else "somewhat disengaged")
        parts.append(
            f"They seemed {eng_word} (engagement {eng * 100:.0f}/100), looking toward the "
            f"camera about {s['eye_contact_pct']:.0f}% of the time."
        )

        expr = s["expressiveness"]
        expr_word = ("very expressive and animated" if expr > 0.55 else
                     "moderately expressive" if expr > 0.3 else
                     "reserved, holding a mostly neutral face")
        parts.append(f"Their expression was {expr_word}.")

        bpm = s["blink_rate"]
        if bpm > 32:
            parts.append(f"A fast blink rate ({bpm:.0f}/min) hints at some nervousness.")
        elif bpm > 0:
            parts.append(f"Blink rate ({bpm:.0f}/min) stayed in a calm, normal range.")

        moments = self.emotional_moments()
        if moments:
            ms = "; ".join(f"{fmt_time(t)} ({emo}, {v * 100:.0f}%)" for t, emo, v in moments)
            parts.append(f"Their most emotionally expressive moments were at {ms}.")
        else:
            parts.append("No strong emotional peaks stood out; affect stayed steady throughout.")

        return " ".join(parts)

    def print_report(self) -> None:
        s = self.summary()
        if not s:
            print("No data captured (was a face visible?).")
            return
        print("\n" + "=" * 56)
        print("  INTERVIEW SENTIMENT REPORT")
        print("=" * 56)
        print(f"  Duration         : {fmt_time(s['duration_s'])}  ({s['frames']} frames)")
        print(f"  Dominant emotion : {s['dominant_emotion']}")
        print(f"  Engagement       : {s['mean_engagement'] * 100:5.1f} / 100")
        print(f"  Attentiveness    : {s['attentiveness'] * 100:5.1f} / 100")
        print(f"  Eye contact      : {s['eye_contact_pct']:5.1f} %")
        print(f"  Positivity       : {s['positivity'] * 100:5.1f} / 100")
        print(f"  Composure        : {s['composure'] * 100:5.1f} / 100")
        print(f"  Expressiveness   : {s['expressiveness'] * 100:5.1f} / 100")
        print(f"  Mean valence     : {s['mean_valence']:+.2f}   (-1 neg .. +1 pos)")
        print(f"  Blink rate       : {s['blink_rate']:5.1f} / min")
        print("-" * 56)
        print("  How they felt:")
        for line in _wrap(self.describe(), 52):
            print(f"    {line}")
        moments = self.emotional_moments()
        if moments:
            print("-" * 56)
            print("  Peak emotional moments:")
            for t, emo, v in moments:
                print(f"    {fmt_time(t):>6}   {emo:<9} {v * 100:3.0f}%")
        print("=" * 56)

    # ----------------------------------------------------------------- files
    def save(self) -> tuple[Path, Path] | None:
        if not self.t:
            return None
        REPORTS_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        png = self._save_plot(REPORTS_DIR / f"interview_{stamp}.png")
        txt = REPORTS_DIR / f"interview_{stamp}.txt"
        s = self.summary()
        with open(txt, "w") as fh:
            fh.write("INTERVIEW SENTIMENT REPORT\n")
            fh.write("=" * 40 + "\n\n")
            for k, v in s.items():
                fh.write(f"{k:18s}: {v}\n")
            fh.write("\nDescription:\n" + self.describe() + "\n\n")
            fh.write("Peak emotional moments:\n")
            for t, emo, val in self.emotional_moments():
                fh.write(f"  {fmt_time(t)}  {emo}  {val * 100:.0f}%\n")
        return png, txt

    def _save_plot(self, out: Path) -> Path:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

        overall = [e.overall for e in self.engagement]
        ax1.plot(self.t, overall, color="#1d9e75", lw=2)
        ax1.fill_between(self.t, overall, color="#1d9e75", alpha=0.15)
        ax1.plot(self.t, self.valence, color="#378add", lw=1.2, alpha=0.8, label="valence")
        ax1.axhline(0, color="#888", lw=0.6, ls=":")
        ax1.set_ylim(-1, 1)
        ax1.set_ylabel("engagement / valence")
        ax1.set_title("Engagement & valence over time")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(alpha=0.2)

        series = {e: [f.get(e, 0.0) for f in self.emotions] for e in EMOTIONS}
        ax2.stackplot(self.t, *series.values(), labels=series.keys(), alpha=0.85)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("emotion mix")
        ax2.set_xlabel("time (s)")
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=7,
                   fontsize=8, frameon=False)
        ax2.set_title("Emotion distribution over time  (dashed = peak moments)")

        # annotate the strongest emotional moments on both panels.
        for t, emo, v in self.emotional_moments():
            for ax in (ax1, ax2):
                ax.axvline(t, color="#993355", ls="--", lw=1, alpha=0.7)
            ax2.annotate(f"{emo}\n{fmt_time(t)}", xy=(t, 0.96), ha="center", va="top",
                         fontsize=8, color="#993355",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#993355", alpha=0.8))

        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return out

    # ----------------------------------------------------------- window image
    def render_image(self, graph_path: Path | None = None, width: int = 880) -> np.ndarray:
        """Render the full report (stats + description + moments + graph) into a
        single BGR image, for display in a native OpenCV 'Session Report' window."""
        import cv2

        BG, FG, DIM, ACC, HOT = (30, 28, 26), (238, 238, 238), (165, 165, 160), \
            (120, 200, 140), (110, 110, 235)
        s = self.summary()
        if not s:
            canvas = np.full((180, width, 3), BG, np.uint8)
            cv2.putText(canvas, "No session data captured (was a face visible?)",
                        (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FG, 1, cv2.LINE_AA)
            return canvas

        pad, line_h = 24, 24
        desc_lines = _wrap(self.describe(), 92)
        moments = self.emotional_moments()
        stats = [
            ("Duration", fmt_time(s["duration_s"])),
            ("Dominant emotion", s["dominant_emotion"]),
            ("Engagement", f"{s['mean_engagement'] * 100:.0f}/100"),
            ("Eye contact", f"{s['eye_contact_pct']:.0f}%"),
            ("Positivity", f"{s['positivity'] * 100:.0f}/100"),
            ("Expressiveness", f"{s['expressiveness'] * 100:.0f}/100"),
            ("Composure", f"{s['composure'] * 100:.0f}/100"),
            ("Blink rate", f"{s['blink_rate']:.0f}/min"),
        ]

        graph = None
        gh = 0
        if graph_path and Path(graph_path).exists():
            g = cv2.imread(str(graph_path))
            if g is not None:
                gh = int(width * g.shape[0] / g.shape[1])
                graph = cv2.resize(g, (width, gh))

        rows = (len(stats) + 1) // 2
        text_h = (pad + 30 + 12) + rows * line_h + 12 + 26 + len(desc_lines) * 22 \
            + 12 + 26 + max(1, len(moments)) * 22 + pad
        canvas = np.full((text_h + gh, width, 3), BG, np.uint8)

        def put(txt, x, y, color=FG, scale=0.5, thick=1):
            cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        thick, cv2.LINE_AA)

        y = pad + 18
        put("SESSION REPORT", pad, y, ACC, 0.8, 2)
        y += 18
        cv2.line(canvas, (pad, y), (width - pad, y), (70, 68, 64), 1)
        y += 24

        col2 = width // 2
        for i in range(rows):
            for j, idx in enumerate((i, i + rows)):
                if idx >= len(stats):
                    continue
                lbl, val = stats[idx]
                x = pad if j == 0 else col2
                put(f"{lbl}:", x, y, DIM)
                put(val, x + 165, y, FG)
            y += line_h
        y += 12

        put("How they felt", pad, y, ACC, 0.6, 2)
        y += 26
        for ln in desc_lines:
            put(ln, pad, y, FG)
            y += 22
        y += 12

        put("Peak emotional moments", pad, y, ACC, 0.6, 2)
        y += 26
        if moments:
            for t, emo, v in moments:
                put(f"{fmt_time(t):>6}   {emo:<10} {v * 100:3.0f}%", pad, y, HOT)
                y += 22
        else:
            put("steady affect — no strong peaks", pad, y, DIM)
            y += 22

        if graph is not None:
            canvas[text_h:text_h + gh, 0:width] = graph
        return canvas


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
