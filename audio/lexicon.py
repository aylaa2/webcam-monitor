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
                    "coding", "programming", "software", "code", "rust", "go", "php",
                    "ruby", "kotlin", "swift", "scala", "algorithm", "algorithms",
                    "object-oriented", "functional"},
    "Data / ML": {"machine", "learning", "ml", "ai", "data", "statistics",
                  "analytics", "model", "models", "neural", "pandas", "numpy",
                  "tensorflow", "pytorch", "regression", "classification", "nlp",
                  "deep", "dataset", "training", "scikit"},
    "Web / Frontend": {"react", "angular", "vue", "html", "css", "frontend",
                       "node", "django", "flask", "api", "rest", "backend",
                       "fullstack", "javascript", "nextjs", "express", "graphql"},
    "Cloud / DevOps": {"aws", "azure", "gcp", "cloud", "docker", "kubernetes",
                       "devops", "ci", "cd", "linux", "terraform", "deployment",
                       "pipeline", "ansible", "jenkins", "microservices"},
    "Databases": {"sql", "database", "databases", "postgres", "postgresql",
                  "mysql", "mongodb", "redis", "nosql", "query", "queries", "schema"},
    "Mobile": {"android", "ios", "flutter", "react-native", "mobile", "swiftui"},
    "Security": {"security", "cybersecurity", "encryption", "authentication",
                 "vulnerability", "penetration", "firewall", "oauth"},
    "Testing / QA": {"testing", "test", "tests", "unit", "integration", "pytest",
                     "selenium", "qa", "automation"},
    "Design": {"design", "ui", "ux", "figma", "photoshop", "prototype",
               "wireframe", "illustrator", "branding"},
    "Project Management": {"agile", "scrum", "kanban", "jira", "roadmap",
                           "stakeholder", "sprint", "backlog"},
    "Business": {"marketing", "sales", "finance", "accounting", "budget",
                 "strategy", "revenue", "customer", "seo", "growth"},
}

# Soft-skill keyword map (backup to the semantic classifier).
SOFT_SKILLS: dict[str, set[str]] = {
    "Communication": {"communication", "communicate", "communicated", "present",
                      "presented", "presentation", "explain", "explained",
                      "articulate", "writing", "wrote", "documented"},
    "Teamwork": {"team", "teamwork", "collaborate", "collaborated", "collaboration",
                 "colleagues", "cooperate", "together", "peers"},
    "Leadership": {"lead", "led", "leadership", "manage", "managed", "management",
                   "mentor", "mentored", "ownership", "initiative", "coordinated",
                   "supervised", "guided"},
    "Problem-solving": {"problem", "problems", "solve", "solved", "analytical",
                        "troubleshoot", "debug", "debugged", "critical", "logical"},
    "Adaptability": {"adapt", "adapted", "flexible", "flexibility", "learn",
                     "learned", "quickly", "versatile", "self-taught"},
    "Time management": {"deadline", "deadlines", "prioritize", "prioritized",
                        "organize", "organized", "schedule", "planning", "punctual"},
    "Creativity": {"creative", "creativity", "innovative", "innovation", "idea",
                   "ideas", "brainstorm", "imaginative"},
    "Work ethic": {"hardworking", "dedicated", "reliable", "responsible",
                   "committed", "motivated", "proactive", "diligent"},
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
    soft_skills: dict[str, list[str]] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    # Unicode letters so Romanian diacritics (ă â î ș ț) stay part of the word.
    return re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)


def analyze(text: str) -> WordStats:
    words = tokenize(text)
    n = len(words)
    if n == 0:
        return WordStats()
    fillers = sum(1 for w in words if w in FILLERS)
    pos = sum(1 for w in words if w in POSITIVE)
    neg = sum(1 for w in words if w in NEGATIVE)
    actions = sum(1 for w in words if w in ACTION_VERBS)

    wordset = set(words)
    found_hard: dict[str, list[str]] = {}
    for cat, keys in HARD_SKILLS.items():
        hits = sorted(wordset & keys)
        if hits:
            found_hard[cat] = hits
    found_soft: dict[str, list[str]] = {}
    for cat, keys in SOFT_SKILLS.items():
        hits = sorted(wordset & keys)
        if hits:
            found_soft[cat] = hits

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
        hard_skills=found_hard,
        soft_skills=found_soft,
    )
