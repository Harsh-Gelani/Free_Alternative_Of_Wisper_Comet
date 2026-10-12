"""
Canto GRPO (Group Relative Policy Optimization) Scorer & Alignment Layer
Implements Wispr Flow Canto's reinforcement learning alignment:
- Multi-candidate rollout evaluation
- Reward functions: Acoustic Likelihood, Language Fluency, Disfluency/Filler Penalty,
  Hallucination Suppression, and Short-Utterance Protection.
- Group Relative Advantage calculation: A_i = (R_i - mean(R)) / (std(R) + eps)
- Optimal rollout selection and policy loss computation.
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional


COMMON_FILLERS = {
    "um", "uh", "er", "ah", "umm", "uhh", "err", "ahh",
    "you know", "like", "basically", "literally", "sort of", "kind of"
}


class CantoGRPOScorer:
    """
    GRPO Alignment Evaluator for Canto Speech Model.
    Scores candidate rollouts within a group and selects the optimal transcript
    based on relative advantage.
    """

    def __init__(
        self,
        weight_acoustic: float = 0.35,
        weight_fluency: float = 0.25,
        weight_filler_penalty: float = 0.20,
        weight_hallucination_penalty: float = 0.30,
        weight_short_preservation: float = 0.15,
        custom_vocabulary: Optional[List[str]] = None,
    ):
        self.w_acoustic = weight_acoustic
        self.w_fluency = weight_fluency
        self.w_filler = weight_filler_penalty
        self.w_hallucination = weight_hallucination_penalty
        self.w_short = weight_short_preservation
        self.custom_vocabulary = [w.lower() for w in (custom_vocabulary or [])]

    def compute_reward(self, rollout: Dict[str, Any], ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Computes composite reward for a single candidate rollout.
        Returns detailed breakdown and total score in [-1.0, 1.0] range.
        """
        text = rollout.get("text", "").strip()
        words = text.split()

        # Empty text penalty
        if not text:
            return {
                "r_total": -0.8,
                "r_acoustic": 0.0,
                "r_fluency": 0.0,
                "r_filler": 0.0,
                "r_hallucination": -0.8,
                "r_short": 0.0,
                "r_wer": 0.0,
            }

        # 1. Acoustic Likelihood Reward (normalized from avg_logprob)
        # avg_logprob typically ranges from 0.0 (high confidence) down to -2.0+ (low confidence)
        avg_logprob = rollout.get("avg_logprob", -0.5)
        # Map [-1.5, 0.0] -> [0.0, 1.0]
        r_acoustic = max(0.0, min(1.0, (avg_logprob + 1.5) / 1.5))

        # 2. Fluency & Clean Syntax Reward
        r_fluency = self._evaluate_fluency(text, words)

        # 3. Filler & Disfluency Penalty
        filler_count = self._count_fillers(text)
        # Penalty increases with filler density
        filler_ratio = filler_count / max(len(words), 1)
        r_filler = -min(1.0, filler_ratio * 2.0)

        # 4. Hallucination & Repetition Loop Penalty
        hallucination_penalty = self._evaluate_hallucination(words)
        r_hallucination = -hallucination_penalty

        # 5. Short Utterance Protection
        # Canto excels at short 1-3 word utterances; boost concise confident answers
        r_short = 0.0
        if 1 <= len(words) <= 4:
            if avg_logprob > -0.6 and not hallucination_penalty:
                r_short = 0.4

        # 6. Custom Vocabulary Boost
        vocab_boost = 0.0
        if self.custom_vocabulary:
            lower_text = text.lower()
            for v in self.custom_vocabulary:
                if v in lower_text:
                    vocab_boost += 0.15
        vocab_boost = min(vocab_boost, 0.3)

        # 7. Supervised WER Reward (if ground truth reference provided during training/eval)
        r_wer = 0.0
        if ground_truth:
            wer_score = self._compute_word_error_rate(reference=ground_truth, hypothesis=text)
            r_wer = max(-1.0, 1.0 - 2.0 * wer_score)

        # Weighted Total Reward
        r_total = (
            self.w_acoustic * r_acoustic
            + self.w_fluency * r_fluency
            + self.w_filler * r_filler
            + self.w_hallucination * r_hallucination
            + self.w_short * r_short
            + vocab_boost
        )

        if ground_truth:
            # When training against ground-truth, WER is primary driver
            r_total = 0.5 * r_total + 0.5 * r_wer

        # Clamp to [-1.0, 1.0]
        r_total = max(-1.0, min(1.0, r_total))

        return {
            "r_total": round(r_total, 4),
            "r_acoustic": round(r_acoustic, 4),
            "r_fluency": round(r_fluency, 4),
            "r_filler": round(r_filler, 4),
            "r_hallucination": round(r_hallucination, 4),
            "r_short": round(r_short, 4),
            "r_wer": round(r_wer, 4),
        }

    def evaluate_group(
        self,
        rollouts: List[Dict[str, Any]],
        ground_truth: Optional[str] = None,
        eps: float = 1e-6,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Executes Group Relative Policy Optimization (GRPO) evaluation.
        Computes group mean, standard deviation, and relative advantage for each rollout.
        Returns: (best_rollout, rollouts_with_advantages)
        """
        if not rollouts:
            return {}, []

        scored_rollouts = []
        rewards = []

        for r in rollouts:
            breakdown = self.compute_reward(r, ground_truth=ground_truth)
            total_r = breakdown["r_total"]
            rewards.append(total_r)
            r_copy = dict(r)
            r_copy["reward_breakdown"] = breakdown
            r_copy["reward"] = total_r
            scored_rollouts.append(r_copy)

        # Group statistics
        n = len(rewards)
        mean_r = sum(rewards) / n
        variance = sum((x - mean_r) ** 2 for x in rewards) / n
        std_r = math.sqrt(variance)

        # Relative Advantages: A_i = (R_i - mean) / (std + eps)
        for r_entry in scored_rollouts:
            if std_r > eps:
                advantage = (r_entry["reward"] - mean_r) / (std_r + eps)
            else:
                advantage = r_entry["reward"] - mean_r
            r_entry["advantage"] = round(advantage, 4)

        # Select candidate with highest advantage (or highest reward)
        best_rollout = max(scored_rollouts, key=lambda x: (x["advantage"], x["reward"]))

        return best_rollout, scored_rollouts

    def compute_grpo_loss(
        self,
        candidate_logprobs: List[float],
        ref_logprobs: List[float],
        advantages: List[float],
        clip_eps: float = 0.2,
        beta_kl: float = 0.04,
    ) -> float:
        """
        Computes the GRPO surrogate clipped policy gradient loss:
        L_GRPO = - 1/G * sum( min( ratio * A, clip(ratio, 1-eps, 1+eps) * A ) ) + beta * KL
        """
        g = len(candidate_logprobs)
        if g == 0:
            return 0.0

        losses = []
        for lp_theta, lp_ref, adv in zip(candidate_logprobs, ref_logprobs, advantages):
            ratio = math.exp(lp_theta - lp_ref)
            surr1 = ratio * adv
            surr2 = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio)) * adv
            policy_loss = -min(surr1, surr2)

            # KL divergence penalty: KL(pi_theta || pi_ref) approx (ratio - 1) - log(ratio)
            kl = ratio - 1.0 - (lp_theta - lp_ref)
            total_sample_loss = policy_loss + beta_kl * max(0.0, kl)
            losses.append(total_sample_loss)

        return sum(losses) / g

    def _evaluate_fluency(self, text: str, words: List[str]) -> float:
        """Heuristic language fluency score."""
        if not words:
            return 0.0

        score = 0.7  # Baseline reasonable English

        # Proper casing and terminal punctuation bonus
        if text[0].isupper():
            score += 0.1
        if text[-1] in {".", "!", "?"}:
            score += 0.1

        # Check for ungrammatical symbol clutter
        if re.search(r"[*&^%$#@~`_+=\\|<>]", text):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _count_fillers(self, text: str) -> int:
        """Counts occurrences of conversational fillers."""
        count = 0
        lower = text.lower()
        tokens = re.findall(r"\b[a-zA-Z']+\b", lower)

        for token in tokens:
            if token in COMMON_FILLERS:
                count += 1

        for phrase in ["you know", "sort of", "kind of"]:
            count += lower.count(phrase)

        return count

    def _evaluate_hallucination(self, words: List[str]) -> float:
        """
        Detects repetitive degeneration (e.g. 'Thank you. Thank you. Thank you.')
        Returns penalty in [0.0, 1.0].
        """
        if len(words) < 4:
            return 0.0

        # Check repeating n-grams (bigrams and trigrams)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = set(bigrams)
        repeat_ratio = 1.0 - (len(unique_bigrams) / max(len(bigrams), 1))

        if repeat_ratio > 0.4:
            return 1.0  # Severe repetition loop detected

        # Check exact word repetition
        word_counts = {}
        for w in words:
            w_lower = w.lower()
            word_counts[w_lower] = word_counts.get(w_lower, 0) + 1

        max_rep = max(word_counts.values()) if word_counts else 0
        if max_rep >= 4 and max_rep / len(words) > 0.4:
            return 0.8

        return 0.0

    def _compute_word_error_rate(self, reference: str, hypothesis: str) -> float:
        """Levenshtein word error rate between reference and hypothesis."""
        r = re.findall(r"\b\w+\b", reference.lower())
        h = re.findall(r"\b\w+\b", hypothesis.lower())

        if not r:
            return 1.0 if h else 0.0

        d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
        for i in range(len(r) + 1):
            d[i][0] = i
        for j in range(len(h) + 1):
            d[0][j] = j

        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return float(d[len(r)][len(h)]) / float(len(r))
