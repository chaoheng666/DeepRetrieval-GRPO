import unittest

import torch
from types import SimpleNamespace

from app_config import PromptConfig, RewardConfig
from core.grpo_engine import normalize_advantages, ppo_clipped_objective
from core.reward_func import (
    Rewarder,
    clean_rewritten_query,
    compose_reward,
    compute_bad_format_penalty,
    compute_clean_format_score,
    compute_keyword_preserve,
    compute_length_score,
    compute_locked_term_preserve,
    compute_mrr_at_k,
    compute_overedit_penalty,
    compute_rank_bonus,
    compute_recall_at_k,
    compute_term_preserve,
    compute_unsafe_copy_penalty,
    is_retrieval_ready_query,
    stabilize_generated_rewrite,
)


class AdvantageTests(unittest.TestCase):
    def test_normalize_advantages_zero_std(self):
        adv = normalize_advantages([1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(adv, torch.zeros_like(adv)))

    def test_normalize_advantages_centered(self):
        adv = normalize_advantages([0.0, 1.0, 2.0])
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(adv.std(unbiased=False)), 1.0, places=5)


class PpoClipTests(unittest.TestCase):
    def test_clipped_objective_positive_advantage(self):
        logprob_old = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32))
        logprob_new = torch.log(torch.tensor([0.75, 0.25], dtype=torch.float32))
        objective = ppo_clipped_objective(logprob_new, logprob_old, advantage=1.0, clip_range=0.2)
        self.assertEqual(objective.shape[0], 2)
        self.assertTrue(torch.allclose(objective, torch.tensor([1.2, 0.5]), atol=1e-5))


class RewardMathTests(unittest.TestCase):
    def test_compute_mrr_at_k_hit(self):
        mrr, rank = compute_mrr_at_k(["D1", "D2", "D3"], {"D3"}, topk=10)
        self.assertAlmostEqual(mrr, 1.0 / 3.0)
        self.assertEqual(rank, 3)

    def test_compute_recall_at_k_hit(self):
        recall, hit_count, total = compute_recall_at_k(["D1", "D2", "D3"], {"D2", "D9"}, topk=3)
        self.assertAlmostEqual(recall, 0.5)
        self.assertEqual(hit_count, 1)
        self.assertEqual(total, 2)

    def test_term_preserve_blends_keyword_and_locked_terms(self):
        term_preserve, number_preserve, acronym_preserve, negation_preserve = compute_term_preserve(
            "COPD home treatment without oxygen 2024",
            "copd treatment oxygen",
        )
        keyword_preserve = compute_keyword_preserve(
            "COPD home treatment without oxygen 2024",
            "copd treatment oxygen",
        )
        locked_term_preserve, _, _, _ = compute_locked_term_preserve(
            "COPD home treatment without oxygen 2024",
            "copd treatment oxygen",
        )

        self.assertEqual(number_preserve, 0.0)
        self.assertEqual(acronym_preserve, 1.0)
        self.assertEqual(negation_preserve, 0.0)
        self.assertAlmostEqual(keyword_preserve, 2.0 / 3.0)
        self.assertAlmostEqual(locked_term_preserve, 1.0 / 3.0)
        self.assertAlmostEqual(term_preserve, 0.5)

    def test_term_preserve_still_varies_when_no_locked_terms_exist(self):
        term_preserve, number_preserve, acronym_preserve, negation_preserve = compute_term_preserve(
            "best budget gaming laptop",
            "gaming laptop deals",
        )
        keyword_preserve = compute_keyword_preserve(
            "best budget gaming laptop",
            "gaming laptop deals",
        )
        locked_term_preserve, _, _, _ = compute_locked_term_preserve(
            "best budget gaming laptop",
            "gaming laptop deals",
        )

        self.assertAlmostEqual(keyword_preserve, 0.5)
        self.assertEqual(locked_term_preserve, 1.0)
        self.assertAlmostEqual(term_preserve, 0.75)
        self.assertEqual(number_preserve, 1.0)
        self.assertEqual(acronym_preserve, 1.0)
        self.assertEqual(negation_preserve, 1.0)

    def test_keyword_preserve_filters_question_words_and_stopwords(self):
        self.assertEqual(
            compute_keyword_preserve(
                "what are symptoms of anemia in women",
                "anemia symptoms women",
            ),
            1.0,
        )

    def test_length_score_piecewise_profile(self):
        cfg = RewardConfig()
        self.assertEqual(compute_length_score("one", cfg), 0.0)
        self.assertAlmostEqual(compute_length_score("one two three", cfg), 2.0 / 3.0)
        self.assertEqual(compute_length_score("one two three four", cfg), 1.0)
        self.assertEqual(compute_length_score("one two three four five six seven eight nine ten eleven twelve", cfg), 1.0)
        self.assertEqual(
            compute_length_score(
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen",
                cfg,
            ),
            0.125,
        )
        self.assertEqual(
            compute_length_score(
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
                cfg,
            ),
            0.0,
        )

    def test_bad_format_penalty_is_continuous(self):
        cfg = RewardConfig()
        self.assertEqual(compute_bad_format_penalty("", cfg), 1.0)
        self.assertEqual(compute_bad_format_penalty("first line\nsecond line", cfg), 0.5)
        self.assertEqual(compute_bad_format_penalty("because this query is better", cfg), 0.5)
        self.assertAlmostEqual(compute_bad_format_penalty("best budget gaming laptop 2024", RewardConfig(format_max_tokens=3)), 0.06)
        self.assertGreater(compute_bad_format_penalty("天气 预报 北京 明天", cfg), 0.0)
        self.assertEqual(compute_clean_format_score(0.0), 1.0)
        self.assertEqual(compute_clean_format_score(0.2), 0.0)

    def test_unsafe_copy_penalty_only_applies_to_non_retrieval_ready_queries(self):
        self.assertEqual(
            compute_unsafe_copy_penalty(
                "what is windows media player amr files",
                "what is windows media player amr files",
            ),
            1.0,
        )
        self.assertEqual(
            compute_unsafe_copy_penalty(
                "windows media player amr files",
                "windows media player amr files",
            ),
            0.0,
        )

    def test_rank_bonus_profile(self):
        self.assertEqual(compute_rank_bonus(1), 1.0)
        self.assertEqual(compute_rank_bonus(3), 0.8)
        self.assertEqual(compute_rank_bonus(5), 0.5)
        self.assertEqual(compute_rank_bonus(10), 0.3)
        self.assertEqual(compute_rank_bonus(20), 0.15)
        self.assertEqual(compute_rank_bonus(50), 0.05)
        self.assertEqual(compute_rank_bonus(80), 0.0)
        self.assertEqual(compute_rank_bonus(None), 0.0)

    def test_overedit_penalty_uses_keyword_threshold(self):
        cfg = RewardConfig()
        self.assertAlmostEqual(compute_overedit_penalty(0.10, cfg), 0.30)
        self.assertEqual(compute_overedit_penalty(0.45, cfg), 0.0)

    def test_total_reward_formula(self):
        cfg = RewardConfig()
        total = compose_reward(
            mrr=0.5,
            recall=0.4,
            recall_dense=0.7,
            term_preserve=0.8,
            length_score=1.0,
            clean_format=1.0,
            bad_format_penalty=0.25,
            unsafe_copy_penalty=1.0,
            cfg=cfg,
        )
        expected = (
            0.40 * 0.5
            + 0.20 * 0.4
            + 0.15 * 0.7
            + 0.10 * 0.8
            + 0.08 * 1.0
            + 0.07 * 1.0
            - 0.15 * 0.25
            - 0.08 * 1.0
        )
        self.assertAlmostEqual(total, expected)

    def test_top20_delta_reward_formula(self):
        cfg = RewardConfig(
            reward_mode="top20_delta",
            w_mrr=0.55,
            w_recall=0.20,
            w_recall_dense=0.15,
            w_rank_bonus=0.10,
            w_bad_format=0.18,
            w_unsafe_copy=0.12,
            w_overedit=0.10,
        )
        total = compose_reward(
            mrr=0.25,
            recall=0.5,
            recall_dense=0.75,
            rank_bonus=0.30,
            orig_mrr=0.05,
            orig_recall=0.25,
            orig_recall_aux=0.50,
            orig_rank_bonus=0.05,
            bad_format_penalty=0.20,
            unsafe_copy_penalty=1.0,
            overedit_penalty=0.10,
            cfg=cfg,
        )
        expected = (
            0.55 * 0.20
            + 0.20 * 0.25
            + 0.15 * 0.25
            + 0.10 * 0.25
            - 0.18 * 0.20
            - 0.12 * 1.0
            - 0.10 * 0.10
        )
        self.assertAlmostEqual(total, expected)

    def test_retrieval_ready_query_detection(self):
        self.assertTrue(is_retrieval_ready_query("windows media player amr files"))
        self.assertFalse(is_retrieval_ready_query("what is windows media player amr files"))


class QueryCleaningTests(unittest.TestCase):
    def test_clean_rewritten_query_prefers_high_overlap_complete_candidate(self):
        raw = (
            "Androgen receptor definition\n\n"
            "Rewritten Query: Definition of the androgen receptor protein\n\n"
            "Rewritten Query: Information on the"
        )
        cleaned = clean_rewritten_query(raw, source_query="Androgen receptor define")
        self.assertEqual(cleaned, "Androgen receptor definition")

    def test_clean_rewritten_query_handles_single_line_quotes(self):
        raw = '   "causes   of climate change 2024 report"   '
        cleaned = clean_rewritten_query(raw)
        self.assertEqual(cleaned, "causes of climate change 2024 report")

    def test_clean_rewritten_query_uses_marker_next_line(self):
        raw = (
            "Search Query:\n"
            "  best budget gaming laptop 2024\n"
            "\n"
            "Explanation: keep under $1000"
        )
        cleaned = clean_rewritten_query(raw, source_query="best budget gaming laptop 2024")
        self.assertEqual(cleaned, "best budget gaming laptop 2024")

    def test_clean_rewritten_query_discards_prompt_template_leakage(self):
        raw = (
            "finderscope\n\n"
            "Example\n"
            "User query: what is a finderscope\n"
            "Better BM25 query:"
        )
        cleaned = clean_rewritten_query(raw, source_query="what is a finderscope")
        self.assertEqual(cleaned, "finderscope")

    def test_stabilize_generated_rewrite_only_falls_back_when_cleaning_is_empty(self):
        cfg = RewardConfig()
        prompt_cfg = PromptConfig(min_terms=3, max_terms=12, fallback_mode="conservative")

        empty_record = stabilize_generated_rewrite(
            "",
            source_query="best laptop under 1000 2024",
            guardrail_cfg=prompt_cfg,
            reward_cfg=cfg,
        )
        self.assertTrue(empty_record.fallback_to_original)
        self.assertEqual(empty_record.final_query, "best laptop under 1000 2024")
        self.assertIn("empty_after_clean", empty_record.fallback_reasons)

        soft_constraint_record = stabilize_generated_rewrite(
            "best laptop under 1000",
            source_query="best laptop under 1000 2024",
            guardrail_cfg=prompt_cfg,
            reward_cfg=cfg,
        )
        self.assertFalse(soft_constraint_record.fallback_to_original)
        self.assertEqual(soft_constraint_record.final_query, "best laptop under 1000")

    def test_stabilize_generated_rewrite_keeps_explanation_like_output_for_soft_scoring(self):
        cfg = RewardConfig()
        prompt_cfg = PromptConfig(min_terms=3, max_terms=11, fallback_mode="balanced")
        record = stabilize_generated_rewrite(
            "because this query is better",
            source_query="best budget gaming laptop 2024",
            guardrail_cfg=prompt_cfg,
            reward_cfg=cfg,
        )
        self.assertFalse(record.fallback_to_original)
        self.assertEqual(record.final_query, "because this query is better")
        self.assertGreater(record.raw_format_penalty, 0.0)


class RewarderConsistencyTests(unittest.TestCase):
    def test_score_cleans_query_before_search(self):
        rewarder = Rewarder.__new__(Rewarder)
        seen: dict[str, object] = {}

        def _search_docids(query: str):
            seen["search_query"] = query
            return ["D1"]

        def _score_one(
            qid: str,
            cleaned_query: str,
            hits_docids: list[str],
            source_query: str | None,
            *,
            original_baseline=None,
        ):
            seen["score_query"] = cleaned_query
            seen["score_hits"] = list(hits_docids)
            seen["score_qid"] = qid
            seen["score_source"] = source_query
            seen["score_baseline"] = original_baseline
            return cleaned_query

        rewarder._search_docids = _search_docids  # type: ignore[attr-defined]
        rewarder._score_one = _score_one  # type: ignore[attr-defined]
        rewarder._get_original_baseline = lambda qid, source_query: None  # type: ignore[attr-defined]

        raw = "Search Query:\n  best budget gaming laptop 2024\nExplanation: keep concise"
        scored = rewarder.score("q1", raw, source_query="best budget gaming laptop 2024")

        self.assertEqual(scored, "best budget gaming laptop 2024")
        self.assertEqual(seen["search_query"], "best budget gaming laptop 2024")
        self.assertEqual(seen["score_query"], "best budget gaming laptop 2024")
        self.assertEqual(seen["score_hits"], ["D1"])
        self.assertEqual(seen["score_qid"], "q1")
        self.assertEqual(seen["score_source"], "best budget gaming laptop 2024")
        self.assertIsNone(seen["score_baseline"])

    def test_score_batch_cleans_queries_before_batch_search(self):
        rewarder = Rewarder.__new__(Rewarder)
        seen: dict[str, object] = {}

        def _search_docids_batch(queries: list[str]):
            seen["search_queries"] = list(queries)
            return [["D1"], ["D2"]]

        def _score_one(
            qid: str,
            cleaned_query: str,
            hits_docids: list[str],
            source_query: str | None,
            *,
            original_baseline=None,
        ):
            return {
                "qid": qid,
                "query": cleaned_query,
                "hits": list(hits_docids),
                "source": source_query,
                "baseline": original_baseline,
            }

        rewarder._search_docids_batch = _search_docids_batch  # type: ignore[attr-defined]
        rewarder._score_one = _score_one  # type: ignore[attr-defined]
        rewarder._get_original_baseline = lambda qid, source_query: None  # type: ignore[attr-defined]

        raw_queries = [
            "Search Query:\n  best budget gaming laptop 2024\nExplanation: keep concise",
            "mastoidectomy\n\nUser query: what is the capital of the united states\nBetter BM25",
        ]
        scored = rewarder.score_batch("q1", raw_queries, source_query="best budget gaming laptop 2024")

        self.assertEqual(
            seen["search_queries"],
            ["best budget gaming laptop 2024", "mastoidectomy"],
        )
        self.assertEqual(scored[0]["query"], "best budget gaming laptop 2024")
        self.assertEqual(scored[1]["query"], "mastoidectomy")

    def test_original_baseline_cache_reuses_cached_lookup(self):
        rewarder = Rewarder.__new__(Rewarder)
        rewarder.original_baseline_cache = {}
        rewarder.qrels = {}
        rewarder.mrr_k = 20
        rewarder.recall_k = 20
        rewarder.recall_dense_k = 50
        rewarder.cfg = RewardConfig(reward_mode="top20_delta")
        seen_queries: list[tuple[str, int | None]] = []

        def _search_docids(query: str, *, k: int | None = None):
            seen_queries.append((query, k))
            return ["D1"]

        def _build_original_baseline(qid: str, source_query: str, hits_docids: list[str]):
            del qid, hits_docids
            return SimpleNamespace(
                query=source_query,
                mrr=0.0,
                recall=0.0,
                recall_aux=0.0,
                rank_bonus=0.0,
                hit_rank=None,
                retrieved_relevant_count=0,
                relevant_total=0,
            )

        rewarder._search_docids = _search_docids  # type: ignore[attr-defined]
        rewarder._build_original_baseline = _build_original_baseline  # type: ignore[attr-defined]

        first = rewarder._get_original_baseline("q1", "source query")
        second = rewarder._get_original_baseline("q1", "source query")

        self.assertEqual(first.query, "source query")
        self.assertEqual(second.query, "source query")
        self.assertEqual(seen_queries, [("source query", None)])


if __name__ == "__main__":
    unittest.main()
