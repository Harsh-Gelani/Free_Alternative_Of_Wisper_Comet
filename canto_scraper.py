"""
Canto Web Scraper & Conversational Dataset Builder
Collects free dictation text, technical speech, and real-world conversational transcripts.
Zero API cost - uses public web sources, Wikipedia, and procedural conversational generation.
"""

import os
import re
import json
import random
import requests
from typing import List, Dict, Any, Optional


PUBLIC_WIKI_API = "https://en.wikipedia.org/w/api.php"

# Real-world dictation templates across 5 key persona domains:
# 1. Executive / Email dictation
# 2. Software Engineering / Technical coding
# 3. Meeting notes & summaries
# 4. Casual messaging & Slack chats
# 5. Whisper / Short command dictations
REAL_WORLD_SEEDS = [
    # Domain: Executive & Email
    {
        "clean": "Please send the quarterly financial update to the board by Friday morning.",
        "spoken": "Um, please send the quarterly financial update to the board by Thursday, actually make that Friday morning.",
        "domain": "email",
        "has_correction": True,
        "has_fillers": True,
    },
    {
        "clean": "Let us schedule a follow-up discussion on Tuesday at 3 PM to review the contract.",
        "spoken": "Uh, let us schedule a follow-up discussion on Tuesday at 2 PM, sorry I mean 3 PM to review the contract.",
        "domain": "email",
        "has_correction": True,
        "has_fillers": True,
    },
    {
        "clean": "Thank you for the quick turnaround on the slide deck. It looks great.",
        "spoken": "Thanks for the like quick turnaround on the slide deck. It it looks great.",
        "domain": "email",
        "has_correction": False,
        "has_fillers": True,
    },
    # Domain: Engineering & Coding
    {
        "clean": "Refactor the database connection pool in FastAPI and deploy to Kubernetes.",
        "spoken": "Refactor the database connection pool in FastAPI and deploy to Docker, scratch that deploy to Kubernetes.",
        "domain": "coding",
        "has_correction": True,
        "has_fillers": False,
    },
    {
        "clean": "Check the Redis cache latency metrics before merging the pull request.",
        "spoken": "Um, check the Redis cache latency metrics before merging the PR, you know.",
        "domain": "coding",
        "has_correction": False,
        "has_fillers": True,
    },
    {
        "clean": "Wispr Flow runs Canto for real-time speech recognition with low latency.",
        "spoken": "Wisper flow runs Canto for like real-time speech recognition with ultra low latency.",
        "domain": "technical",
        "has_correction": False,
        "has_fillers": True,
    },
    {
        "clean": "Add unit tests for the authentication middleware and handle token expiry.",
        "spoken": "Add unit tests for auth middleware, actually make sure to handle token expiry too.",
        "domain": "coding",
        "has_correction": True,
        "has_fillers": False,
    },
    # Domain: Meeting & Notetaker
    {
        "clean": "Action item for Sarah: draft the product specification by end of week.",
        "spoken": "Action item for Sarah, uh, draft the product specification by end of week.",
        "domain": "meeting",
        "has_correction": False,
        "has_fillers": True,
    },
    {
        "clean": "The team agreed to migrate the frontend to Next.js App Router.",
        "spoken": "So basically the team agreed to migrate the frontend to Next.js App Router.",
        "domain": "meeting",
        "has_correction": False,
        "has_fillers": True,
    },
    # Domain: Short Command / Whisper
    {
        "clean": "Stop recording.",
        "spoken": "Stop recording.",
        "domain": "command",
        "is_whisper": True,
        "has_correction": False,
        "has_fillers": False,
    },
    {
        "clean": "Open settings.",
        "spoken": "Open settings.",
        "domain": "command",
        "is_whisper": True,
        "has_correction": False,
        "has_fillers": False,
    },
    {
        "clean": "Approve pull request.",
        "spoken": "Approve PR.",
        "domain": "command",
        "is_whisper": True,
        "has_correction": False,
        "has_fillers": False,
    },
]


class CantoDataScraper:
    """
    Collects and synthesizes clean reference transcripts and conversational speech samples.
    """

    def __init__(self, output_dir: str = "./data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def scrape_wikipedia_sentences(self, topics: Optional[List[str]] = None, max_sentences: int = 50) -> List[str]:
        """
        Scrapes real sentences from Wikipedia articles (free public domain text)
        on technical and everyday topics.
        """
        if topics is None:
            topics = [
                "Speech_recognition",
                "Machine_learning",
                "Cloud_computing",
                "Software_engineering",
                "Artificial_intelligence",
            ]

        scraped_sentences = []
        headers = {"User-Agent": "CantoSpeechModelScraper/1.0 (free research tool)"}

        for topic in topics:
            if len(scraped_sentences) >= max_sentences:
                break
            try:
                params = {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": topic,
                    "format": "json",
                }
                resp = requests.get(PUBLIC_WIKI_API, params=params, headers=headers, timeout=6.0)
                if resp.status_code == 200:
                    data = resp.json()
                    pages = data.get("query", {}).get("pages", {})
                    for page_id, page_data in pages.items():
                        extract = page_data.get("extract", "")
                        # Split into clean sentences
                        raw_sents = re.split(r"[.!?]+", extract)
                        for s in raw_sents:
                            cleaned = s.strip()
                            # Keep natural dictation length (8 to 25 words)
                            word_count = len(cleaned.split())
                            if 8 <= word_count <= 25 and not cleaned.startswith("=="):
                                scraped_sentences.append(cleaned)
                                if len(scraped_sentences) >= max_sentences:
                                    break
            except Exception as e:
                print(f"[Scraper] Wikipedia fetch for '{topic}' failed: {e}. Using local corpus.")

        return scraped_sentences

    def inject_conversational_artifacts(self, clean_sentence: str) -> Dict[str, Any]:
        """
        Synthesizes real-world speech disfluency into clean text:
        - Injects fillers ('um', 'uh', 'you know', 'like')
        - Injects self-corrections ('actually', 'scratch that')
        - Injects stuttering
        """
        words = clean_sentence.split()
        if len(words) < 5:
            return {
                "clean": clean_sentence,
                "spoken": clean_sentence,
                "domain": "general",
                "has_fillers": False,
                "has_correction": False,
            }

        spoken_words = list(words)
        has_filler = False
        has_correction = False

        # 50% chance of leading filler
        if random.random() < 0.6:
            filler = random.choice(["Um, ", "Uh, ", "So basically, "])
            spoken_words[0] = filler + spoken_words[0].lower()
            has_filler = True

        # 35% chance of mid-sentence filler
        if random.random() < 0.35 and len(spoken_words) > 6:
            idx = random.randint(2, len(spoken_words) - 3)
            spoken_words.insert(idx, random.choice(["like,", "you know,", "sort of"]))
            has_filler = True

        # 25% chance of stutter
        if random.random() < 0.25 and len(spoken_words) > 4:
            stutter_idx = random.randint(1, len(spoken_words) - 2)
            target = spoken_words[stutter_idx]
            if target.isalpha():
                spoken_words[stutter_idx] = f"{target} {target}"

        spoken_text = " ".join(spoken_words)

        # 30% chance of self-correction
        if random.random() < 0.30 and len(words) > 8:
            cut_pt = len(words) // 2
            correction_trigger = random.choice([", actually ", ", no wait ", ", scratch that "])
            distractor = random.choice(["tomorrow morning", "the staging server", "the old version"])
            spoken_text = f"{' '.join(words[:cut_pt])} {distractor}{correction_trigger}{' '.join(words[cut_pt:])}"
            has_correction = True

        return {
            "clean": clean_sentence.strip(),
            "spoken": spoken_text.strip(),
            "domain": "synthesized",
            "has_fillers": has_filler,
            "has_correction": has_correction,
        }

    def build_dataset(self, target_count: int = 40) -> str:
        """
        Compiles complete paired dataset from seeds and scraped web text.
        Saves to ./data/canto_dataset.json.
        """
        print(f"[Scraper] Building dataset with target {target_count} samples...")
        dataset = list(REAL_WORLD_SEEDS)

        # Scrape additional text from Wikipedia
        needed = max(0, target_count - len(dataset))
        if needed > 0:
            print(f"[Scraper] Scraping {needed} sentences from public sources...")
            wiki_sents = self.scrape_wikipedia_sentences(max_sentences=needed)
            for s in wiki_sents:
                entry = self.inject_conversational_artifacts(s)
                dataset.append(entry)

        # Add additional synthetic templates if needed
        while len(dataset) < target_count:
            base = random.choice(REAL_WORLD_SEEDS)
            dataset.append(self.inject_conversational_artifacts(base["clean"]))

        output_path = os.path.join(self.output_dir, "canto_dataset.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset[:target_count], f, indent=2)

        print(f"[Scraper] Saved {len(dataset[:target_count])} training samples to {output_path}")
        return output_path


if __name__ == "__main__":
    scraper = CantoDataScraper()
    path = scraper.build_dataset(target_count=30)
    print("Dataset generation complete:", path)
