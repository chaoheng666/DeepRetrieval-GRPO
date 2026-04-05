import unittest

from retrieval_eval.core import compute_reward, get_or_create_doc_id


class ComputeRewardTests(unittest.TestCase):
    def test_hit_at_rank_1(self):
        qrels = {"Q1": ["D1"]}
        results = [("D1", 10.0), ("D2", 9.0)]
        self.assertEqual(compute_reward("Q1", results, qrels), 1.0)

    def test_hit_at_rank_k(self):
        qrels = {"Q1": ["D3"]}
        results = [("D1", 10.0), ("D2", 9.0), ("D3", 8.0)]
        self.assertAlmostEqual(compute_reward("Q1", results, qrels), 1.0 / 3.0)

    def test_no_hit(self):
        qrels = {"Q1": ["D4"]}
        results = [("D1", 10.0), ("D2", 9.0), ("D3", 8.0)]
        self.assertEqual(compute_reward("Q1", results, qrels), 0.0)

    def test_missing_qid(self):
        qrels = {"Q2": ["D1"]}
        results = [("D1", 10.0)]
        self.assertEqual(compute_reward("Q1", results, qrels), 0.0)


class DocIdMappingTests(unittest.TestCase):
    def test_mapping_is_stable(self):
        text_to_doc = {}
        doc_to_text = {}
        first = get_or_create_doc_id("  retrieval    rocks ", text_to_doc, doc_to_text)
        second = get_or_create_doc_id("retrieval rocks", text_to_doc, doc_to_text)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(len(text_to_doc), 1)
        self.assertEqual(len(doc_to_text), 1)

    def test_empty_text_is_ignored(self):
        text_to_doc = {}
        doc_to_text = {}
        doc_id = get_or_create_doc_id("   ", text_to_doc, doc_to_text)

        self.assertIsNone(doc_id)
        self.assertEqual(text_to_doc, {})
        self.assertEqual(doc_to_text, {})


if __name__ == "__main__":
    unittest.main()
