import unittest

from cachetools import FIFOCache, PartitionedCache


def even_odd(key):
    """Two-partition example: map keys to partitions by parity."""
    return key % 2


class CountingPartition:
    """Partition function wrapper that counts its calls."""

    def __init__(self, func=even_odd):
        self.func = func
        self.count = 0

    def __call__(self, key):
        self.count += 1
        return self.func(key)


class PartitionedCacheTest(unittest.TestCase):
    def test_two_partitions_isolation(self):
        # A = partition 0 (even keys), B = partition 1 (odd keys)
        cache = PartitionedCache({0: 2, 1: 2}, even_odd)

        cache[1] = "b1"
        cache[3] = "b2"
        self.assertEqual(2, len(cache))

        # fill up A, then exceed A's capacity
        cache[0] = "a1"
        cache[2] = "a2"
        cache[4] = "a3"

        # only A's least recently used item (key 0) was evicted
        self.assertNotIn(0, cache)
        self.assertEqual("a2", cache[2])
        self.assertEqual("a3", cache[4])
        # B's items are retained
        self.assertEqual("b1", cache[1])
        self.assertEqual("b2", cache[3])
        self.assertEqual(4, len(cache))

    def test_partition_function_called_once_per_operation(self):
        partition = CountingPartition()
        cache = PartitionedCache({0: 2, 1: 2}, partition)

        cache[1] = "a"
        self.assertEqual(1, partition.count)
        cache[1] = "b"  # update existing key
        self.assertEqual(2, partition.count)
        cache[1]
        self.assertEqual(3, partition.count)
        1 in cache
        self.assertEqual(4, partition.count)
        cache.get(1)
        self.assertEqual(5, partition.count)
        cache.setdefault(1)
        self.assertEqual(6, partition.count)
        cache.pop(1)
        self.assertEqual(7, partition.count)
        cache[3] = "c"
        self.assertEqual(8, partition.count)
        del cache[3]
        self.assertEqual(9, partition.count)

    def test_resize_evicts_per_partition_order(self):
        cache = PartitionedCache({0: 3, 1: 3}, even_odd)

        cache[0] = "a"
        cache[2] = "b"
        cache[4] = "c"
        cache[1] = "x"
        cache[3] = "y"
        cache[5] = "z"
        self.assertEqual(6, len(cache))

        # lower B's (odd) quota: evict in B's own LRU order
        cache[1]  # touch key 1, so 3 is least recently used
        cache.resize({0: 3, 1: 2})
        self.assertEqual({0: 3, 1: 2}, cache.partition_sizes)
        self.assertNotIn(3, cache)
        self.assertEqual("x", cache[1])
        self.assertEqual("z", cache[5])
        # A (even) is completely unaffected
        self.assertEqual("a", cache[0])
        self.assertEqual("b", cache[2])
        self.assertEqual("c", cache[4])

        # lower A's quota: evict in A's own LRU order
        cache[0]  # touch key 0, so 2 is least recently used
        cache.resize({0: 2, 1: 2})
        self.assertNotIn(2, cache)
        self.assertEqual("a", cache[0])
        self.assertEqual("c", cache[4])
        self.assertEqual("x", cache[1])
        self.assertEqual("z", cache[5])

    def test_resize_generates_new_config(self):
        cache = PartitionedCache({0: 2, 1: 2}, even_odd)
        sizes = cache.partition_sizes
        self.assertEqual({0: 2, 1: 2}, sizes)

        cache.resize({0: 4, 1: 1})
        # the previously returned config is not mutated
        self.assertEqual({0: 2, 1: 2}, sizes)
        self.assertEqual({0: 4, 1: 1}, cache.partition_sizes)
        self.assertEqual(5, cache.maxsize)

    def test_resize_add_and_drop_partitions(self):
        cache = PartitionedCache({0: 2}, even_odd)
        cache[0] = "a"
        cache[2] = "b"

        # add a new partition
        cache.resize({0: 2, 1: 2})
        self.assertEqual({0: 2, 1: 2}, cache.partition_sizes)
        self.assertEqual("a", cache[0])
        self.assertEqual("b", cache[2])
        cache[1] = "c"
        self.assertEqual("c", cache[1])

        # drop partition 0 along with its items
        cache.resize({1: 2})
        self.assertEqual(1, len(cache))
        self.assertEqual({1: 2}, cache.partition_sizes)
        self.assertEqual("c", cache[1])
        cache[3] = "d"
        self.assertEqual("d", cache[3])

    def test_resize_negative_maxsize(self):
        cache = PartitionedCache({0: 2}, lambda key: 0)
        with self.assertRaises(ValueError):
            cache.resize({0: -1})

    def test_totals_match_partitions(self):
        cache = PartitionedCache({0: 2, 1: 3}, even_odd, getsizeof=lambda v: v)

        self.assertEqual(5, cache.maxsize)
        self.assertEqual(0, cache.currsize)

        cache[0] = 2  # A: currsize 2
        cache[1] = 1  # B: currsize 1
        cache[3] = 2  # B: currsize 3
        self.assertEqual(5, cache.currsize)
        self.assertEqual(3, len(cache))

        cache[5] = 1  # exceeds B's quota, evicts key 1 (size 1)
        self.assertEqual(5, cache.currsize)
        self.assertEqual(3, len(cache))
        self.assertNotIn(1, cache)
        self.assertEqual(2, cache[0])
        self.assertEqual(2, cache[3])
        self.assertEqual(1, cache[5])

    def test_getsizeof_oversized_item_no_partial_state(self):
        cache = PartitionedCache({0: 2, 1: 2}, even_odd, getsizeof=lambda v: v)

        cache[0] = 2
        cache[1] = 1
        self.assertEqual(3, cache.currsize)

        # a single item larger than its partition is rejected...
        with self.assertRaises(ValueError):
            cache[2] = 3
        # ...and leaves no partial state behind
        self.assertEqual(2, len(cache))
        self.assertEqual(3, cache.currsize)
        self.assertEqual(2, cache[0])
        self.assertEqual(1, cache[1])

        # same for updates of existing keys
        with self.assertRaises(ValueError):
            cache[0] = 3
        self.assertEqual(2, cache[0])
        self.assertEqual(3, cache.currsize)

    def test_cache_interface(self):
        cache = PartitionedCache({0: 2, 1: 2}, even_odd)

        cache.update({0: "a", 1: "b"})
        self.assertEqual("a", cache.get(0))
        self.assertIsNone(cache.get(2))
        self.assertEqual("d", cache.get(2, "d"))

        self.assertEqual("a", cache.setdefault(0))
        self.assertEqual("c", cache.setdefault(2, "c"))

        self.assertEqual("b", cache.pop(1))
        self.assertIsNone(cache.pop(1, None))
        with self.assertRaises(KeyError):
            cache.pop(1)

        self.assertEqual({"a", "c"}, set(cache.values()))
        self.assertEqual({0, 2}, set(cache))
        self.assertEqual(2, len(cache))

        key, value = cache.popitem()
        self.assertIn(key, {0, 2})
        self.assertEqual(1, len(cache))
        cache.popitem()
        with self.assertRaises(KeyError):
            cache.popitem()

        cache[1] = "x"
        cache.clear()
        self.assertEqual(0, len(cache))
        self.assertEqual(0, cache.currsize)

    def test_missing(self):
        class DefaultCache(PartitionedCache):
            def __missing__(self, key):
                self[key] = key
                return key

        cache = DefaultCache({0: 1, 1: 1}, even_odd)

        self.assertEqual(1, cache[1])
        self.assertEqual(2, cache[2])
        self.assertEqual(2, len(cache))
        self.assertEqual(4, cache[4])  # evicts 2, keeps 1
        self.assertEqual(1, cache[1])
        self.assertNotIn(2, cache)

    def test_unknown_partition(self):
        cache = PartitionedCache({0: 2}, lambda key: key % 2)
        with self.assertRaises(KeyError):
            cache[1] = "a"
        with self.assertRaises(KeyError):
            cache[1]
        self.assertNotIn(1, cache)

    def test_cache_factory(self):
        cache = PartitionedCache({0: 2, 1: 2}, even_odd, cache_factory=FIFOCache)
        cache[0] = "a"
        cache[2] = "b"
        cache[0]  # FIFO: access does not refresh
        cache[4] = "c"
        self.assertNotIn(0, cache)
        self.assertEqual("b", cache[2])
        self.assertEqual("c", cache[4])

    def test_default_factory_is_lru(self):
        cache = PartitionedCache({0: 2, 1: 2}, even_odd)
        cache[0] = "a"
        cache[2] = "b"
        cache[0]  # LRU: access refreshes
        cache[4] = "c"
        self.assertEqual("a", cache[0])
        self.assertNotIn(2, cache)

    def test_repr_and_equality(self):
        cache = PartitionedCache({0: 2, 1: 2}, even_odd)
        self.assertTrue(repr(cache).startswith("PartitionedCache("))

        cache[0] = "a"
        other = PartitionedCache({0: 2, 1: 2}, even_odd)
        other[0] = "a"
        self.assertEqual(cache, other)

    def test_is_cache_instance(self):
        from cachetools import Cache

        cache = PartitionedCache({0: 2, 1: 2}, even_odd)
        self.assertIsInstance(cache, Cache)


if __name__ == "__main__":
    unittest.main()
