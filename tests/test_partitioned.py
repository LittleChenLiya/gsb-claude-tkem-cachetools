"""Tests for the PartitionedCache adapter."""

import unittest

from cachetools import FIFOCache, LRUCache, Partition, PartitionedCache


def two_partition_cache(maxsize_a=2, maxsize_b=2, cachecls=LRUCache, getsizeof=None):
    """Build a two-partition cache: keys < 100 -> A, else -> B."""
    calls = []

    def partitionfunc(key):
        calls.append(key)
        return "A" if key < 100 else "B"

    cache = PartitionedCache(
        partitions={
            "A": (maxsize_a, cachecls),
            "B": (maxsize_b, cachecls),
        },
        partitionfunc=partitionfunc,
        getsizeof=getsizeof,
    )
    return cache, calls


class PartitionedCacheTest(unittest.TestCase):
    def test_basic_get_set_delete(self):
        cache, _ = two_partition_cache()
        cache[1] = "a"
        cache[101] = "b"
        self.assertEqual("a", cache[1])
        self.assertEqual("b", cache[101])
        self.assertEqual(2, len(cache))
        del cache[1]
        self.assertNotIn(1, cache)
        self.assertIn(101, cache)

    def test_partition_isolation_a_full_evicts_a_only(self):
        cache, _ = two_partition_cache(maxsize_a=2, maxsize_b=2)
        cache[1] = "a1"
        cache[2] = "a2"
        cache[101] = "b1"
        cache[102] = "b2"
        cache[3] = "a3"
        self.assertEqual(2, len(cache.get_partition("A")))
        self.assertEqual(2, len(cache.get_partition("B")))
        self.assertEqual("b1", cache[101])
        self.assertEqual("b2", cache[102])
        self.assertNotIn(1, cache)
        self.assertIn(2, cache)
        self.assertIn(3, cache)

    def test_partition_isolation_b_full_evicts_b_only(self):
        cache, _ = two_partition_cache(maxsize_a=2, maxsize_b=2)
        cache[1] = "a1"
        cache[2] = "a2"
        cache[101] = "b1"
        cache[102] = "b2"
        cache[103] = "b3"
        self.assertEqual(2, len(cache.get_partition("A")))
        self.assertEqual(2, len(cache.get_partition("B")))
        self.assertEqual("a1", cache[1])
        self.assertEqual("a2", cache[2])
        self.assertNotIn(101, cache)
        self.assertIn(102, cache)
        self.assertIn(103, cache)

    def test_lower_b_quota_evicts_in_b_order(self):
        cache, _ = two_partition_cache(maxsize_a=5, maxsize_b=5, cachecls=FIFOCache)
        for i in range(101, 106):
            cache[i] = "b%d" % i
        self.assertEqual(5, len(cache.get_partition("B")))
        cache.set_partition("B", maxsize=2)
        self.assertEqual(2, len(cache.get_partition("B")))
        self.assertNotIn(101, cache)
        self.assertNotIn(102, cache)
        self.assertNotIn(103, cache)
        self.assertIn(104, cache)
        self.assertIn(105, cache)
        self.assertEqual(0, len(cache.get_partition("A")))

    def test_lower_a_quota_evicts_in_a_order(self):
        cache, _ = two_partition_cache(maxsize_a=5, maxsize_b=5, cachecls=FIFOCache)
        for i in range(1, 6):
            cache[i] = "a%d" % i
        cache.set_partition("A", maxsize=2)
        self.assertEqual(2, len(cache.get_partition("A")))
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertNotIn(3, cache)
        self.assertIn(4, cache)
        self.assertIn(5, cache)

    def test_raise_quota_keeps_items(self):
        cache, _ = two_partition_cache(maxsize_a=2, maxsize_b=2)
        cache[1] = "a1"
        cache[2] = "a2"
        cache.set_partition("A", maxsize=5)
        self.assertEqual(2, len(cache.get_partition("A")))
        self.assertIn(1, cache)
        self.assertIn(2, cache)

    def test_partitionfunc_called_once_per_operation(self):
        cache, calls = two_partition_cache()
        cache[1] = "a"
        self.assertEqual(1, len(calls))
        _ = cache[1]
        self.assertEqual(2, len(calls))
        _ = 1 in cache
        self.assertEqual(3, len(calls))
        cache.get(1)
        self.assertEqual(4, len(calls))
        cache.pop(1)
        self.assertEqual(5, len(calls))
        cache.setdefault(2, "b")
        self.assertEqual(6, len(calls))
        del cache[2]
        self.assertEqual(7, len(calls))
        _ = len(cache)
        _ = list(cache)
        _ = cache.currsize
        _ = cache.maxsize
        cache.clear()
        self.assertEqual(7, len(calls))

    def test_total_size_consistent_with_partitions(self):
        cache, _ = two_partition_cache(maxsize_a=5, maxsize_b=5, getsizeof=len)
        cache[1] = "a" * 2
        cache[2] = "a" * 3
        cache[101] = "b" * 1
        self.assertEqual(
            cache.currsize,
            cache.get_partition("A").currsize + cache.get_partition("B").currsize,
        )
        self.assertEqual(2 + 3 + 1, cache.currsize)
        self.assertEqual(
            cache.maxsize,
            cache.get_partition("A").maxsize + cache.get_partition("B").maxsize,
        )
        cache[3] = "a" * 2
        self.assertEqual(
            cache.currsize,
            cache.get_partition("A").currsize + cache.get_partition("B").currsize,
        )

    def test_oversized_item_leaves_no_partial_state(self):
        cache, _ = two_partition_cache(maxsize_a=2, maxsize_b=2, getsizeof=len)
        cache[1] = "a"
        cache[101] = "b"
        with self.assertRaises(ValueError):
            cache[2] = "a" * 10
        self.assertEqual(1, len(cache.get_partition("A")))
        self.assertIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertIn(101, cache)

    def test_oversized_item_on_quota_lower_evicted(self):
        cache, _ = two_partition_cache(maxsize_a=5, maxsize_b=5, getsizeof=len)
        cache[1] = "a" * 4
        cache[2] = "a" * 1
        self.assertEqual(5, cache.get_partition("A").currsize)
        cache.set_partition("A", maxsize=2)
        self.assertNotIn(1, cache)
        self.assertIn(2, cache)
        self.assertEqual(1, cache.get_partition("A").currsize)

    def test_partition_config_formats(self):
        c1 = PartitionedCache({"A": 1}, lambda k: "A")
        self.assertEqual(1, c1.get_partition("A").maxsize)
        c2 = PartitionedCache({"A": (1, FIFOCache)}, lambda k: "A")
        self.assertEqual(1, c2.get_partition("A").maxsize)
        self.assertIsInstance(c2.get_partition("A"), FIFOCache)
        c3 = PartitionedCache({"A": Partition(1, FIFOCache)}, lambda k: "A")
        self.assertEqual(1, c3.get_partition("A").maxsize)
        self.assertIsInstance(c3.get_partition("A"), FIFOCache)

    def test_unknown_partition_raises(self):
        bad = PartitionedCache({"A": 1}, lambda k: "Z")
        with self.assertRaises(KeyError):
            bad[1] = "x"

    def test_popitem_from_first_nonempty_partition(self):
        cache, _ = two_partition_cache()
        cache[1] = "a"
        cache[101] = "b"
        key, value = cache.popitem()
        self.assertIn(key, (1, 101))
        self.assertEqual(1, len(cache))
        key, value = cache.popitem()
        self.assertEqual(0, len(cache))
        with self.assertRaises(KeyError):
            cache.popitem()

    def test_clear(self):
        cache, _ = two_partition_cache()
        cache[1] = "a"
        cache[101] = "b"
        cache.clear()
        self.assertEqual(0, len(cache))
        self.assertEqual(0, cache.currsize)
        self.assertEqual(0, cache.get_partition("A").currsize)
        self.assertEqual(0, cache.get_partition("B").currsize)

    def test_partition_names(self):
        cache, _ = two_partition_cache()
        self.assertEqual(["A", "B"], cache.partition_names())

    def test_repr(self):
        cache, _ = two_partition_cache()
        cache[1] = "a"
        r = repr(cache)
        self.assertIn("PartitionedCache", r)
        self.assertIn("A", r)
        self.assertIn("B", r)


if __name__ == "__main__":
    unittest.main()