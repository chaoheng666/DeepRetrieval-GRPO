import unittest

from data.curriculum import (
    CurriculumQueryMetadata,
    bucket_for_query,
    filter_curriculum_train_queries,
    sample_curriculum_queries,
)
from data.loader import QueryExample


class CurriculumBucketTests(unittest.TestCase):
    def test_bucket_rules_cover_a_b_c_and_drop(self):
        self.assertEqual(
            bucket_for_query(orig_mrr20=0.10, orig_recall100=1.0, query_text="what are symptoms of anemia in women"),
            "A",
        )
        self.assertEqual(
            bucket_for_query(orig_mrr20=0.40, orig_recall100=1.0, query_text="windows media player amr files"),
            "C",
        )
        self.assertEqual(
            bucket_for_query(orig_mrr20=0.0, orig_recall100=0.0, query_text="symptoms anemia women adults"),
            "B",
        )
        self.assertEqual(
            bucket_for_query(orig_mrr20=0.0, orig_recall100=0.0, query_text="what is it"),
            "DROP",
        )

    def test_bucket_b_rejects_multiline_and_polluted_queries(self):
        self.assertEqual(
            bucket_for_query(
                orig_mrr20=0.0,
                orig_recall100=0.0,
                query_text="symptoms anemia women\nBetter BM25 query:",
            ),
            "DROP",
        )
        self.assertEqual(
            bucket_for_query(
                orig_mrr20=0.0,
                orig_recall100=0.0,
                query_text="assistant: symptoms anemia women",
            ),
            "DROP",
        )

    def test_filter_curriculum_drops_only_drop_bucket(self):
        queries = [
            QueryExample(qid="a", text="query a"),
            QueryExample(qid="b", text="query b"),
            QueryExample(qid="c", text="query c"),
            QueryExample(qid="d", text="query d"),
        ]
        metadata = {
            "a": CurriculumQueryMetadata("a", "query a", 0.1, 0.1, 1.0, 5, "A"),
            "b": CurriculumQueryMetadata("b", "query b", 0.0, 0.0, 0.0, None, "B"),
            "c": CurriculumQueryMetadata("c", "query c", 0.4, 0.5, 1.0, 1, "C"),
            "d": CurriculumQueryMetadata("d", "query d", 0.0, 0.0, 0.0, None, "DROP"),
        }

        kept, counts = filter_curriculum_train_queries(queries, metadata)

        self.assertEqual([query.qid for query in kept], ["a", "b", "c"])
        self.assertEqual(counts["DROP"], 1)


class CurriculumSamplingTests(unittest.TestCase):
    def test_phase1_sampling_prefers_a_bucket(self):
        queries = [
            QueryExample(qid="a1", text="a1"),
            QueryExample(qid="a2", text="a2"),
            QueryExample(qid="b1", text="b1"),
            QueryExample(qid="c1", text="c1"),
        ]
        metadata = {
            "a1": CurriculumQueryMetadata("a1", "a1", 0.1, 0.1, 1.0, 5, "A"),
            "a2": CurriculumQueryMetadata("a2", "a2", 0.1, 0.1, 1.0, 5, "A"),
            "b1": CurriculumQueryMetadata("b1", "b1", 0.0, 0.0, 0.0, None, "B"),
            "c1": CurriculumQueryMetadata("c1", "c1", 0.5, 0.5, 1.0, 1, "C"),
        }

        sampled = sample_curriculum_queries(queries, metadata, phase="phase1", seed=7, epoch=1)

        sampled_buckets = [metadata[query.qid].bucket for query in sampled]
        self.assertEqual(len(sampled), 4)
        self.assertGreaterEqual(sampled_buckets.count("A"), 2)

    def test_sampling_reweights_when_bucket_missing(self):
        queries = [
            QueryExample(qid="a1", text="a1"),
            QueryExample(qid="a2", text="a2"),
            QueryExample(qid="c1", text="c1"),
        ]
        metadata = {
            "a1": CurriculumQueryMetadata("a1", "a1", 0.1, 0.1, 1.0, 5, "A"),
            "a2": CurriculumQueryMetadata("a2", "a2", 0.1, 0.1, 1.0, 5, "A"),
            "c1": CurriculumQueryMetadata("c1", "c1", 0.5, 0.5, 1.0, 1, "C"),
        }

        sampled = sample_curriculum_queries(queries, metadata, phase="phase2", seed=13, epoch=2)

        self.assertEqual(len(sampled), 3)
        self.assertTrue(all(query.qid in metadata for query in sampled))


if __name__ == "__main__":
    unittest.main()
