"""Word-level analysis from the transcript — transparent lexicons, no LLM.

Counts the things an interviewer listens for: filler words, positive/negative
tone, impact ("action") verbs, technical (hard-skill) keywords, and vocabulary
richness. All lists are editable and fully explainable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FILLERS = {
    "um", "uh", "erm", "eh", "ah", "hmm", "like", "basically", "actually",
    "literally", "kinda", "sorta", "stuff", "things", "whatever",
}
POSITIVE = {
    "great", "good", "love", "excited", "confident", "enjoy", "passionate",
    "success", "successful", "achieve", "achieved", "improve", "improved",
    "strong", "happy", "proud", "motivated", "excellent", "effective",
    "collaborate", "growth", "opportunity", "solve", "solved",
}
NEGATIVE = {
    "difficult", "hate", "problem", "fail", "failed", "worried", "nervous",
    "unfortunately", "never", "bad", "hard", "struggle", "struggled", "confused",
    "boring", "weak",
}
# Impact / ownership verbs HR looks for in interviews.
ACTION_VERBS = {
    "led", "lead", "built", "build", "created", "create", "designed", "design",
    "managed", "manage", "developed", "develop", "implemented", "implement",
    "improved", "launched", "launch", "delivered", "deliver", "organized",
    "organised", "solved", "achieved", "increased", "reduced", "optimized",
    "optimised", "automated", "mentored", "coordinated", "owned", "founded",
}
# Hard-skill keyword map: category -> trigger words.
HARD_SKILLS: dict[str, set[str]] = {
    "Programming": {"python", "java", "javascript", "typescript", "c++", "c#",
                    "coding", "programming", "software", "code", "rust", "go", "php"},
    "Data / ML": {"machine", "learning", "ml", "ai", "data", "statistics",
                  "analytics", "model", "neural", "pandas", "numpy", "tensorflow",
                  "pytorch"},
    "Web / Frontend": {"react", "angular", "vue", "html", "css", "frontend",
                       "node", "django", "flask", "api", "backend"},
    "Cloud / DevOps": {"aws", "azure", "gcp", "cloud", "docker", "kubernetes",
                       "devops", "ci", "cd", "linux", "terraform"},
    "Databases": {"sql", "database", "postgres", "mysql", "mongodb", "redis"},
    "Design": {"design", "ui", "ux", "figma", "photoshop", "prototype"},
    "Management": {"agile", "scrum", "project", "leadership", "team", "roadmap",
                   "stakeholder", "kanban"},
    "Business": {"marketing", "sales", "finance", "accounting", "budget",
                 "strategy", "revenue", "customer"},
}


@dataclass
class WordStats:
    total_words: int = 0
    unique_words: int = 0
    fillers: int = 0
    filler_rate: float = 0.0          # fillers per 100 words
    positive: int = 0
    negative: int = 0
    action_verbs: int = 0
    sentiment: float = 0.0            # -1 .. +1 (lexicon-based)
    vocab_richness: float = 0.0       # unique / total
    hard_skills: dict[str, list[str]] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def analyze(text: str) -> WordStats:
    words = tokenize(text)
    n = len(words)
    if n == 0:
        return WordStats()
    fillers = sum(1 for w in words if w in FILLERS)
    pos = sum(1 for w in words if w in POSITIVE)
    neg = sum(1 for w in words if w in NEGATIVE)
    actions = sum(1 for w in words if w in ACTION_VERBS)

    found: dict[str, list[str]] = {}
    wordset = set(words)
    for cat, keys in HARD_SKILLS.items():
        hits = sorted(wordset & keys)
        if hits:
            found[cat] = hits

    sentiment = (pos - neg) / max(1, pos + neg) if (pos + neg) else 0.0
    return WordStats(
        total_words=n,
        unique_words=len(wordset),
        fillers=fillers,
        filler_rate=100.0 * fillers / n,
        positive=pos,
        negative=neg,
        action_verbs=actions,
        sentiment=sentiment,
        vocab_richness=len(wordset) / n,
        hard_skills=found,
    )
