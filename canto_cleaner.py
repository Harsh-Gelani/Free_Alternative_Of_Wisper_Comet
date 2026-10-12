"""
Canto Zero-Edit Disfluency Resolver & Formatting Engine
Cleans natural speech artifacts into polished writing:
- Strips filler tokens ("um", "uh", "you know", "like")
- Resolves mid-sentence self-corrections ("tomorrow, actually Friday" -> "Friday")
- Eliminates stuttering & repetition ("I I want" -> "I want")
- Custom vocabulary injection & tone styling
"""

import re
from typing import List, Dict, Optional, Tuple


# Conversational fillers that can be removed safely
FILLER_PATTERNS = [
    r"\b(so basically)\b",
    r"\b(basically)\b",
    r"\b(literally)\b",
    r"\b(you know)\b",
    r"\b(sort of)\b",
    r"\b(kind of)\b",
    r"\b(um+)\b",
    r"\b(uh+)\b",
    r"\b(er+)\b",
    r"\b(ah+)\b",
]

# Phonetic mishearings and grammar normalization patterns
PHONETIC_PATTERNS = [
    # Wispr Flow variants (Fisper Flu, Whisper Flu, Wisper Flow, Fisper Flow, Whisper Flo, etc.)
    (r"\b(?:[FfWw]isper|[FfWw]hisper|[FfWw]isp[ae]r)\s+(?:[Ff]lu|[Ff]lo|[Ff]low|[Ff]lue)\b", "Wispr Flow"),
    (r"\b(?:[FfWw]isperflow|[FfWw]hisperflow)\b", "Wispr Flow"),
    # Canto variants (Kanto, Cantor, Can-toe, etc.)
    (r"\b(?:[KkCc]anto|[Cc]antor)\b", "Canto"),
    # Common grammar cleanups: "a AI" -> "an AI", "a API" -> "an API"
    (r"\ba\s+AI\b", "an AI"),
    (r"\ba\s+API\b", "an API"),
    (r"\ba\s+ML\b", "an ML"),
    (r"\ba\s+LLM\b", "an LLM"),
    (r"\ba\s+([AaEeIiOoUu][A-Za-z]+)\b", r"an \1"),
]

# Self-correction trigger phrases: phrase before trigger is replaced by phrase after
CORRECTION_TRIGGERS = [
    r",?\s*(?:actually|no wait|scratch that|sorry I mean|I mean|make that|rather)\s*,?\s*",
]


class CantoZeroEditCleaner:
    """
    Post-processing engine transforming raw spoken transcripts into publication-ready text.
    """

    def __init__(
        self,
        custom_dictionary: Optional[Dict[str, str]] = None,
        snippets: Optional[Dict[str, str]] = None,
    ):
        # Maps lowercase misspelled/phonetic words to proper casing (e.g. 'wisperflow' -> 'Wispr Flow')
        self.custom_dictionary = custom_dictionary or {
            "wisper flow": "Wispr Flow",
            "wisperflow": "Wispr Flow",
            "canto": "Canto",
            "kubernetes": "Kubernetes",
            "nextjs": "Next.js",
            "fastapi": "FastAPI",
            "pytorch": "PyTorch",
            "github": "GitHub",
        }
        # Voice snippets: trigger phrases expanded to full templates
        self.snippets = snippets or {}

    def clean_transcript(
        self,
        raw_text: str,
        tone: str = "auto",
        strip_fillers: bool = True,
        resolve_corrections: bool = True,
    ) -> Tuple[str, dict]:
        """
        Runs the full zero-edit cleaning pipeline.
        Returns: (cleaned_text, telemetry)
        """
        metrics = {
            "fillers_removed": 0,
            "corrections_resolved": 0,
            "stutters_removed": 0,
            "snippets_expanded": 0,
            "dictionary_replacements": 0,
        }

        if not raw_text or not raw_text.strip():
            return "", metrics

        text = raw_text.strip()

        # 1. Expand Snippets (e.g. 'my meeting link' -> 'https://meet.google.com/xyz')
        for trigger, replacement in self.snippets.items():
            pattern = re.compile(rf"\b{re.escape(trigger)}\b", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(replacement, text)
                metrics["snippets_expanded"] += 1

        # 2. Resolve Mid-Sentence Self-Corrections
        if resolve_corrections:
            text, corr_count = self._resolve_self_corrections(text)
            metrics["corrections_resolved"] = corr_count

        # 3. Deduplicate Stuttering / Repetitions ("I I think" -> "I think")
        text, stutter_count = self._remove_stutter(text)
        metrics["stutters_removed"] = stutter_count

        # 4. Remove Filler Words ("um", "uh", etc.)
        if strip_fillers:
            text, filler_count = self._remove_fillers(text)
            metrics["fillers_removed"] = filler_count

        # 5. Apply Phonetic Normalization (e.g. 'Fisper Flu' -> 'Wispr Flow', 'Kanto' -> 'Canto')
        for pat, repl in PHONETIC_PATTERNS:
            new_text, n = re.subn(pat, repl, text, flags=re.IGNORECASE)
            if n > 0:
                text = new_text
                metrics["dictionary_replacements"] += n

        # 6. Apply Exact & Fuzzy Custom Dictionary replacements
        text = self._apply_custom_dictionary(text, metrics)

        # 7. Apply Tone Transformation
        text = self._apply_tone(text, tone)

        # 8. Normalize Spacing & Punctuation
        text = self._normalize_punctuation(text)

        return text, metrics

    def _apply_custom_dictionary(self, text: str, metrics: dict) -> str:
        """Applies exact regex and fuzzy dictionary replacement."""
        # 1. Exact replacements
        for lower_word, proper_casing in self.custom_dictionary.items():
            pattern = re.compile(rf"\b{re.escape(lower_word)}\b", re.IGNORECASE)
            new_text, n = pattern.subn(proper_casing, text)
            if n > 0:
                text = new_text
                metrics["dictionary_replacements"] += n

        # 2. Fuzzy matching using rapidfuzz
        try:
            from rapidfuzz import fuzz
            words = text.split()
            i = 0
            new_words = []
            while i < len(words):
                matched = False
                # Try bigrams
                if i < len(words) - 1:
                    clean_bigram = f"{words[i]} {words[i+1]}".strip(".,;:?!")
                    for key, val in self.custom_dictionary.items():
                        if len(key.split()) == 2 and fuzz.ratio(clean_bigram.lower(), key.lower()) >= 72:
                            punc = words[i+1][-1] if words[i+1][-1] in ".,;:?!" else ""
                            new_words.append(val + punc)
                            metrics["dictionary_replacements"] += 1
                            i += 2
                            matched = True
                            break
                # Try single word
                if not matched:
                    clean_single = words[i].strip(".,;:?!")
                    for key, val in self.custom_dictionary.items():
                        if len(key.split()) == 1 and fuzz.ratio(clean_single.lower(), key.lower()) >= 82:
                            punc = words[i][-1] if words[i][-1] in ".,;:?!" else ""
                            new_words.append(val + punc)
                            metrics["dictionary_replacements"] += 1
                            matched = True
                            i += 1
                            break
                if not matched:
                    new_words.append(words[i])
                    i += 1
            text = " ".join(new_words)
        except ImportError:
            pass

        return text

    def _resolve_self_corrections(self, text: str) -> Tuple[str, int]:
        """
        Detects phrases like:
        - "Please schedule for Thursday, actually Friday at 2 PM." -> "Please schedule for Friday at 2 PM."
        - "Please send the invoice by Thursday. Actually, no. Friday morning." -> "Please send the invoice by Friday morning."
        - "Check deployment logs, actually scratch that, check database connection." -> "Check database connection."
        """
        count = 0

        # Pattern 1: Sentence-level correction with preposition:
        # e.g. "by Thursday. Actually, no. Friday morning" -> "by Friday morning"
        prep_corr_pattern = re.compile(
            r"(\b(?:by|at|on|for|to|in|from)\s+)(?:[A-Za-z0-9]+\b[.,;]?\s*)(?:actually[.,;]?\s*(?:no[.,;]?\s*)?|scratch that[.,;]?\s*|make that[.,;]?\s*)([A-Za-z0-9\s]+)",
            re.IGNORECASE
        )
        new_text, n = prep_corr_pattern.subn(r"\1\2", text)
        if n > 0:
            count += n
            text = new_text

        # Pattern 2: Clause replacement: "Check X, actually scratch that, check Y"
        clause_pattern = re.compile(
            r"(?:^|[.!?]\s*)(?:[^.!?]+)[,.]\s*(?:actually scratch that|scratch that|no wait|sorry I mean)[,.]?\s*([^.!?]+)",
            re.IGNORECASE
        )
        new_text, n = clause_pattern.subn(r"\1", text)
        if n > 0:
            count += n
            text = new_text

        # Pattern 3: Comma separated correction: "Thursday, actually Friday"
        simple_pattern = re.compile(
            r"(\b[A-Za-z0-9]+)\s*,\s*(?:actually|make that|rather)\s+([A-Za-z0-9]+)",
            re.IGNORECASE
        )
        new_text, n = simple_pattern.subn(r"\2", text)
        if n > 0:
            count += n
            text = new_text

        return text, count

    def _remove_stutter(self, text: str) -> Tuple[str, int]:
        """Removes consecutive duplicate words like 'the the' or 'I I'."""
        stutter_pattern = re.compile(r"\b([a-zA-Z]+)\s+\1\b", re.IGNORECASE)
        count = 0

        while True:
            new_text, num = stutter_pattern.subn(r"\1", text)
            if num == 0:
                break
            count += num
            text = new_text

        return text, count

    def _remove_fillers(self, text: str) -> Tuple[str, int]:
        """Removes verbal fillers without breaking sentence structure."""
        count = 0
        for pattern_str in FILLER_PATTERNS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            new_text, num = pattern.subn("", text)
            if num > 0:
                count += num
                text = new_text

        # Clean dangling 'like' at beginning or surrounded by commas
        # e.g. "Like, we need to do this" -> "We need to do this"
        new_text, num = re.subn(r"^\s*like,\s*", "", text, flags=re.IGNORECASE)
        count += num
        text = new_text

        new_text, num = re.subn(r",\s*like,\s*", ", ", text, flags=re.IGNORECASE)
        count += num
        text = new_text

        return text, count

    def _apply_tone(self, text: str, tone: str) -> str:
        """Applies requested tone transformations."""
        tone = tone.lower().strip()

        if tone == "bullet points" or tone == "bullets":
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
            if len(sentences) > 1:
                return "\n".join(f"• {s}." for s in sentences)
            return f"• {text}"

        elif tone == "concise":
            # Strip preamble phrases like 'I think that', 'In my opinion', 'Could you please'
            preambles = [
                r"^\s*I think (?:that)?\s*",
                r"^\s*In my opinion,?\s*",
                r"^\s*Basically,?\s*",
                r"^\s*Just wanted to say that\s*",
            ]
            for p in preambles:
                text = re.sub(p, "", text, flags=re.IGNORECASE)

        elif tone == "formal":
            # Replace informal contractions
            contractions = {
                r"\bcan't\b": "cannot",
                r"\bwon't\b": "will not",
                r"\bdon't\b": "do not",
                r"\bgonna\b": "going to",
                r"\bwanna\b": "want to",
                r"\byeah\b": "yes",
                r"\byep\b": "yes",
                r"\bnope\b": "no",
            }
            for c_pat, formal_rep in contractions.items():
                text = re.sub(c_pat, formal_rep, text, flags=re.IGNORECASE)

        elif tone == "code":
            # Format inline code identifiers (camelCase, snake_case)
            words = text.split()
            formatted_words = []
            for w in words:
                clean_w = w.strip(".,;:?!")
                if ("_" in clean_w or any(c.isupper() for c in clean_w[1:])) and clean_w.isalnum():
                    formatted_words.append(w.replace(clean_w, f"`{clean_w}`"))
                else:
                    formatted_words.append(w)
            text = " ".join(formatted_words)

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Ensures proper capitalization and punctuation spacing."""
        # Strip leading punctuation left behind by removed fillers (e.g. ", please send...")
        text = re.sub(r"^[,.:;?!-]+\s*", "", text).strip()

        # Clean duplicate or awkward consecutive punctuation (e.g. ",.", ", .", ",,")
        text = re.sub(r"[,;]+\s*([.!?])", r"\1", text)
        text = re.sub(r"[,;]{2,}", ",", text)
        text = re.sub(r"\.{2,}", ".", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Fix space before punctuation: "hello , world ." -> "hello, world."
        text = re.sub(r"\s+([,.:;?!])", r"\1", text)

        # Fix missing space after punctuation: "hello,world" -> "hello, world"
        text = re.sub(r"([,.:;?!])([a-zA-Z])", r"\1 \2", text)

        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Capitalize after sentence boundaries
        text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

        # Add period at end if missing punctuation
        if text and text[-1] not in {".", "!", "?", "`", ":", ";"}:
            text += "."

        return text
