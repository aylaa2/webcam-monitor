"""Transcript-level analysis — all rule/lexicon based, no LLM.

  - find_quotes()    : pull the exact sentences where given words were spoken
  - key_concepts()   : the salient terms the candidate talked about
  - background()     : sentences that describe experience / background
  - detect_star()    : does the answer follow Situation-Task-Action-Result?
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from audio.lexicon import ACTION_VERBS, tokenize

# Stopwords (English + a few common Romanian) so key-concept extraction surfaces
# meaningful content words in either language.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "at", "for",
    "with", "is", "am", "are", "was", "were", "be", "been", "i", "you", "he", "she",
    "it", "we", "they", "my", "your", "our", "this", "that", "these", "those", "as",
    "if", "then", "than", "have", "has", "had", "do", "did", "not", "no", "yes",
    "can", "could", "would", "should", "will", "just", "really", "very", "more",
    "some", "any", "all", "about", "like", "from", "by", "me", "him", "her", "them",
    "what", "when", "where", "which", "who", "how", "there", "here", "also", "because",
    # Romanian
    "si", "de", "la", "in", "cu", "pe", "un", "o", "este", "sunt", "am", "eu", "sa",
    "ca", "mai", "care", "pentru", "din", "nu", "se", "ai", "al", "ale", "lui", "meu",
    "mea", "foarte", "dar", "sau", "ne", "le", "te", "ma", "asta", "acest", "fost",
}

STAR_CUES: dict[str, set[str]] = {
    "Situation": {"the situation", "there was a", "i was working", "at the time",
                  "we had a", "the problem was", "faced with", "the challenge",
                  "context", "in my previous", "at my previous", "during", "once when",
                  "background of", "the issue was"},
    "Task": {"task", "my job was", "i was responsible", "responsible for", "the goal",
             "i had to", "needed to", "objective", "my role was", "asked to",
             "supposed to", "required to", "i was assigned"},
    "Action": {"i decided", "i implemented", "i built", "i created", "i developed",
               "i designed", "i led", "i organized", "my approach", "i started by",
               "first i", "then i", "i used", "to solve this", "the steps", "i chose"},
    "Result": {"as a result", "the result", "the outcome", "i achieved", "we achieved",
               "increased", "reduced", "improved", "led to", "successfully", "in the end",
               "percent", "%", "saved", "grew", "the impact", "ended up"},
}

BACKGROUND_CUES = {
    "i worked", "i have", "years", "year", "experience", "my background", "i studied",
    "i graduated", "graduated", "degree", "university", "college", "i built",
    "i developed", "i was a", "my role", "internship", "i led", "i managed",
    "i started", "currently", "previously", "i am a", "specialized", "focused on",
    "my job", "i work", "i used", "i learned", "self taught", "certified", "company",
}


@dataclass
class Star:
    present: dict[str, bool] = field(default_factory=dict)
    evidence: dict[str, list[str]] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return sum(self.present.values()) / 4.0 if self.present else 0.0


def find_quotes(segments, wordset: set[str], limit: int = 4) -> list[tuple[float, str]]:
    """Segments whose tokens intersect `wordset`, with their start time."""
    out: list[tuple[float, str]] = []
    for s in segments:
        toks = set(tokenize(s.text))
        if toks & wordset:
            out.append((s.start, s.text))
            if len(out) >= limit:
                break
    return out


def find_phrase_quotes(segments, phrases: set[str], limit: int = 6) -> list[tuple[float, str]]:
    """Segments whose lowercased text contains any of the cue phrases."""
    out: list[tuple[float, str]] = []
    for s in segments:
        low = s.text.lower()
        if any(p in low for p in phrases):
            out.append((s.start, s.text))
            if len(out) >= limit:
                break
    return out


def key_concepts(text: str, hard_skills: dict[str, list[str]], top: int = 12
                 ) -> list[tuple[str, int]]:
    toks = [t for t in tokenize(text) if len(t) > 2 and t not in STOPWORDS]
    counts = Counter(toks)
    # promote detected hard-skill keywords to the front.
    skill_words = {w for kws in hard_skills.values() for w in kws}
    ranked: list[tuple[str, int]] = []
    seen: set[str] = set()
    for w in sorted(skill_words, key=lambda w: -counts.get(w, 1)):
        ranked.append((w, counts.get(w, 1)))
        seen.add(w)
    for w, c in counts.most_common():
        if w in seen:
            continue
        ranked.append((w, c))
        if len(ranked) >= top:
            break
    return ranked[:top]


def background(segments, limit: int = 6) -> list[tuple[float, str]]:
    return find_phrase_quotes(segments, BACKGROUND_CUES, limit)


def detect_star(text: str, action_verb_boost: bool = True) -> Star:
    low = " " + text.lower() + " "
    star = Star()
    for comp, cues in STAR_CUES.items():
        hits = [c for c in cues if c in low]
        present = len(hits) > 0
        if comp == "Action" and action_verb_boost and not present:
            # an answer full of action verbs counts as describing actions
            if sum(1 for w in tokenize(text) if w in ACTION_VERBS) >= 2:
                present = True
                hits = ["(uses multiple action verbs)"]
        star.present[comp] = present
        star.evidence[comp] = hits
    return star
