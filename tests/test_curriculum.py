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
    def test_phase1_sampling_uses_80_10_10_mix(self):
        queries = [QueryExample(qid=f"a{i}", text=f"a{i}") for i in range(1, 5)]
        queries += [QueryExample(qid=f"b{i}", text=f"b{i}") for i in range(1, 4)]
        queries += [QueryExample(qid=f"c{i}", text=f"c{i}") for i in range(1, 4)]
        metadata = {
            query.qid: CurriculumQueryMetadata(
                query.qid,
                query.text,
                0.1 if query.qid.startswith("a") else 0.0,
                0.1 if query.qid.startswith("a") else 0.0,
                1.0,
                5,
                "A" if query.qid.startswith("a") else "B" if query.qid.startswith("b") else "C",
            )
            for query in queries
        }

        sampled = sample_curriculum_queries(queries, metadata, phase="phase1", seed=7, epoch=1)
        sampled_buckets = [metadata[query.qid].bucket for query in sampled]

        self.assertEqual(len(sampled), 10)
        self.assertEqual(sampled_buckets.count("A"), 8)
        self.assertEqual(sampled_buckets.count("B"), 1)
        self.assertEqual(sampled_buckets.count("C"), 1)

    def test_phase2_sampling_uses_65_25_10_mix(self):
        queries = [QueryExample(qid=f"a{i}", text=f"a{i}") for i in range(1, 5)]
        queries += [QueryExample(qid=f"b{i}", text=f"b{i}") for i in range(1, 4)]
        queries += [QueryExample(qid=f"c{i}", text=f"c{i}") for i in range(1, 4)]
        metadata = {
            query.qid: CurriculumQueryMetadata(
                query.qid,
                query.text,
                0.1 if query.qid.startswith("a") else 0.0,
                0.1 if query.qid.startswith("a") else 0.0,
                1.0,
                5,
                "A" if query.qid.startswith("a") else "B" if query.qid.startswith("b") else "C",
            )
            for query in queries
        }

        sampled = sample_curriculum_queries(queries, metadata, phase="phase2", seed=13, epoch=2)
        sampled_buckets = [metadata[query.qid].bucket for query in sampled]

        self.assertEqual(len(sampled), 10)
        self.assertEqual(sampled_buckets.count("A"), 7)
        self.assertEqual(sampled_buckets.count("B"), 2)
        self.assertEqual(sampled_buckets.count("C"), 1)


if __name__ == "__main__":
    unittest.main()
